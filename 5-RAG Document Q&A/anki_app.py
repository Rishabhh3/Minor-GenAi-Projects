import datetime as dt
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import genanki
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def stable_anki_id(seed_text: str, offset: int = 0) -> int:
	"""Generate stable positive 31-bit IDs required by genanki."""
	digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
	return (int(digest[:8], 16) + offset) % 2147483647


def normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def parse_tags(tag_text: str) -> list:
	tags = []
	for token in re.split(r"[\s,]+", tag_text.strip()):
		token = token.strip()
		if token:
			tags.append(token)
	return tags


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
	return HuggingFaceEmbeddings(model_name=model_name)


def load_pdf_pages(uploaded_pdf) -> list:
	with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
		temp_pdf.write(uploaded_pdf.getvalue())
		temp_path = temp_pdf.name

	try:
		loader = PyPDFLoader(temp_path)
		pages = loader.load()
	finally:
		os.unlink(temp_path)

	return pages


def split_pages(pages: list, chunk_size: int, chunk_overlap: int) -> list:
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	chunks = splitter.split_documents(pages)
	for idx, chunk in enumerate(chunks):
		chunk.metadata["chunk_id"] = idx
	return chunks


def choose_representative_chunks(vectorstore: FAISS, max_chunks: int) -> list:
	seed_queries = [
		"core concepts and important definitions",
		"key facts and exam-relevant points",
		"formulas, steps, and procedures",
		"contrasts, comparisons, and exceptions",
		"high-yield summary",
	]
	selected = []
	seen_ids = set()

	per_query_k = max(3, min(8, max_chunks // 2))
	for query in seed_queries:
		docs = vectorstore.similarity_search(query, k=per_query_k)
		for doc in docs:
			cid = doc.metadata.get("chunk_id")
			if cid is None or cid in seen_ids:
				continue
			seen_ids.add(cid)
			selected.append(doc)
			if len(selected) >= max_chunks:
				return selected

	# Fallback: if retrieval did not yield enough variety.
	if len(selected) < max_chunks:
		fallback = vectorstore.similarity_search("important concepts", k=max_chunks)
		for doc in fallback:
			cid = doc.metadata.get("chunk_id")
			if cid is None or cid in seen_ids:
				continue
			seen_ids.add(cid)
			selected.append(doc)
			if len(selected) >= max_chunks:
				break

	return selected


def extract_json_array(raw_text: str) -> list:
	cleaned = raw_text.strip()
	if cleaned.startswith("```"):
		cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
		cleaned = re.sub(r"```$", "", cleaned).strip()

	try:
		parsed = json.loads(cleaned)
		return parsed if isinstance(parsed, list) else []
	except json.JSONDecodeError:
		pass

	match = re.search(r"\[[\s\S]*\]", raw_text)
	if not match:
		return []
	try:
		parsed = json.loads(match.group(0))
		return parsed if isinstance(parsed, list) else []
	except json.JSONDecodeError:
		return []


def ensure_cloze_markup(text: str) -> str:
	if re.search(r"\{\{c\d+::", text):
		return text
	return text


def generate_cards_for_chunk(
	llm: ChatGroq,
	chunk_text: str,
	supporting_context: str,
	cards_per_chunk: int,
	source_page: int,
	card_type: str,
	card_instructions: str,
) -> list:
	if card_type == "Cloze Deletion":
		prompt = f"""You are an expert study coach. Generate cloze deletion flashcards from the source.

Rules:
1) Return ONLY a valid JSON array.
2) Each item must contain keys: text, extra, tags.
3) In text, include at least one valid Anki cloze pattern like {{c1::important phrase}}.
4) Keep one core concept per card.
5) Avoid duplicate cards.
6) Keep output exam-ready and high-yield.

Target count: {cards_per_chunk}
Source page: {source_page}

Additional author instructions:
{card_instructions}

Primary excerpt:
---
{chunk_text}
---

Additional retrieved context:
---
{supporting_context}
---
"""
	else:
		prompt = f"""You are an expert study coach. Generate high-quality flashcards from the source.

Rules:
1) Return ONLY a valid JSON array.
2) Each item must contain keys: question, answer, extra, tags.
3) Keep each question precise and unambiguous.
4) Keep each answer concise but complete (1-4 sentences).
5) Avoid duplicates or near-duplicates.
6) Prefer conceptual understanding over trivia.
7) If the content has formulas, include at least one formula-based card when possible.
8) Keep output exam-ready and high-yield.

Target count: {cards_per_chunk}
Source page: {source_page}

Additional author instructions:
{card_instructions}

Primary excerpt:
---
{chunk_text}
---

Additional retrieved context:
---
{supporting_context}
---
"""

	response = llm.invoke(prompt)
	raw = response.content if hasattr(response, "content") else str(response)
	records = extract_json_array(raw)

	cards = []
	for rec in records:
		if not isinstance(rec, dict):
			continue
		extra = normalize_space(str(rec.get("extra", "")))
		tags = normalize_space(str(rec.get("tags", "")))

		if card_type == "Cloze Deletion":
			text = ensure_cloze_markup(str(rec.get("text", "")).strip())
			if len(text) < 24 or "{{c" not in text:
				continue
			cards.append(
				{
					"card_type": card_type,
					"text": text,
					"question": "",
					"answer": "",
					"extra": extra,
					"tags": tags,
					"source_page": source_page,
				}
			)
		else:
			question = normalize_space(str(rec.get("question", "")))
			answer = normalize_space(str(rec.get("answer", "")))
			if len(question) < 12 or len(answer) < 8:
				continue
			if question.lower() == answer.lower():
				continue
			cards.append(
				{
					"card_type": card_type,
					"text": "",
					"question": question,
					"answer": answer,
					"extra": extra,
					"tags": tags,
					"source_page": source_page,
				}
			)
	return cards


def deduplicate_cards(cards: list) -> list:
	unique = {}
	for card in cards:
		if card.get("card_type") == "Cloze Deletion":
			key = normalize_space(card.get("text", "")).lower()
		else:
			key = normalize_space(card.get("question", "")).lower()
		if key not in unique:
			unique[key] = card
	return list(unique.values())


def cards_to_csv_bytes(cards: list) -> bytes:
	output = io.StringIO()
	output.write("card_type,question,answer,text,extra,tags,source_page\n")
	for card in cards:
		row = [
			card.get("card_type", "Basic Q/A").replace('"', '""'),
			card["question"].replace('"', '""'),
			card["answer"].replace('"', '""'),
			card.get("text", "").replace('"', '""'),
			card["extra"].replace('"', '""'),
			card["tags"].replace('"', '""'),
			str(card["source_page"]),
		]
		output.write(
			f'"{row[0]}","{row[1]}","{row[2]}","{row[3]}","{row[4]}","{row[5]}","{row[6]}"\n'
		)
	return output.getvalue().encode("utf-8")


def cards_to_apkg_bytes(cards: list, deck_name: str, card_type: str, anki_config: dict) -> bytes:
	basic_qfmt = anki_config["basic_qfmt"]
	basic_afmt = anki_config["basic_afmt"]
	reverse_qfmt = anki_config["reverse_qfmt"]
	reverse_afmt = anki_config["reverse_afmt"]
	cloze_qfmt = anki_config["cloze_qfmt"]
	cloze_afmt = anki_config["cloze_afmt"]
	model_css = anki_config["model_css"]
	model_name = anki_config["model_name"]
	deck_description = anki_config["deck_description"]
	include_source = anki_config["include_source"]
	include_extra = anki_config["include_extra"]
	include_reverse = anki_config["include_reverse"]
	global_tags = parse_tags(anki_config["global_tags"])

	if card_type == "Cloze Deletion":
		model = genanki.Model(
			model_id=stable_anki_id(deck_name, offset=21),
			name=f"{model_name} (Cloze)",
			model_type=genanki.Model.CLOZE,
			fields=[
				{"name": "Text"},
				{"name": "Extra"},
				{"name": "Source"},
			],
			templates=[
				{
					"name": "Cloze Card",
					"qfmt": cloze_qfmt,
					"afmt": cloze_afmt,
				}
			],
			css=model_css,
		)
	else:
		templates = [
			{
				"name": "Card 1",
				"qfmt": basic_qfmt,
				"afmt": basic_afmt,
			}
		]
		if include_reverse:
			templates.append(
				{
					"name": "Card 2 (Reverse)",
					"qfmt": reverse_qfmt,
					"afmt": reverse_afmt,
				}
			)

		model = genanki.Model(
			model_id=stable_anki_id(deck_name, offset=11),
			name=f"{model_name} (Basic)",
			fields=[
				{"name": "Question"},
				{"name": "Answer"},
				{"name": "Extra"},
				{"name": "Source"},
			],
			templates=templates,
			css=model_css,
		)

	deck = genanki.Deck(
		deck_id=stable_anki_id(deck_name, offset=99),
		name=deck_name,
	)
	if deck_description.strip():
		deck.description = deck_description

	for idx, card in enumerate(cards):
		if card_type == "Cloze Deletion":
			note_seed = card.get("text", "")
			note_fields = [
				card.get("text", ""),
				card["extra"] if include_extra else "",
				f"Source page: {card['source_page']}" if include_source else "",
			]
		else:
			note_seed = card["question"]
			note_fields = [
				card["question"],
				card["answer"],
				card["extra"] if include_extra else "",
				f"Source page: {card['source_page']}" if include_source else "",
			]

		card_tags = [t for t in card["tags"].split() if t]
		merged_tags = sorted(set(global_tags + card_tags))

		note = genanki.Note(
			model=model,
			fields=note_fields,
			guid=hashlib.md5(f"{deck_name}-{idx}-{note_seed}".encode("utf-8")).hexdigest()[:10],
			tags=merged_tags,
		)
		deck.add_note(note)

	with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as temp_apkg:
		temp_path = temp_apkg.name

	try:
		genanki.Package(deck).write_to_file(temp_path)
		with open(temp_path, "rb") as file:
			return file.read()
	finally:
		os.unlink(temp_path)


st.set_page_config(page_title="PDF to Anki RAG Generator", layout="wide")
st.title("PDF to Anki Flashcards (RAG + LLM)")
st.caption(
	"Upload a PDF, build retrieval context, generate high-quality flashcards, and download an Anki package (.apkg)."
)

with st.sidebar:
	st.header("Model Settings")
	groq_api_key = st.text_input(
		"GROQ_API_KEY",
		value=os.getenv("GROQ_API_KEY", ""),
		type="password",
	)
	model_name = st.selectbox(
		"Groq model",
		["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
		index=0,
	)

	st.header("Chunking")
	chunk_size = st.slider("Chunk size", min_value=800, max_value=2400, value=1300, step=100)
	chunk_overlap = st.slider("Chunk overlap", min_value=80, max_value=500, value=180, step=20)

	st.header("Anki Controls")
	deck_name_override = st.text_input("Deck name override", value="")
	model_name_override = st.text_input("Anki note model name", value="RAG Flashcard Model")
	deck_description = st.text_area("Deck description", value="", height=80)
	global_tags = st.text_input("Global tags (space or comma separated)", value="rag pdf")
	include_reverse = st.checkbox("Basic: include reverse card", value=False)
	include_extra = st.checkbox("Include Extra field", value=True)
	include_source = st.checkbox("Include Source page field", value=True)
	card_instructions = st.text_area(
		"Extra generation instructions",
		value="Focus on high-yield exam facts and avoid trivia.",
		height=80,
	)

	with st.expander("Templates and Styling", expanded=False):
		basic_qfmt = st.text_area("Basic front template", value="{{Question}}", height=80)
		basic_afmt = st.text_area(
			"Basic back template",
			value="{{FrontSide}}<hr id=answer>{{Answer}}<br><br><small>{{Extra}}</small><br><small>{{Source}}</small>",
			height=120,
		)
		reverse_qfmt = st.text_area("Reverse front template", value="{{Answer}}", height=80)
		reverse_afmt = st.text_area(
			"Reverse back template",
			value="{{FrontSide}}<hr id=answer>{{Question}}<br><br><small>{{Extra}}</small><br><small>{{Source}}</small>",
			height=120,
		)
		cloze_qfmt = st.text_area("Cloze front template", value="{{cloze:Text}}", height=80)
		cloze_afmt = st.text_area(
			"Cloze back template",
			value="{{cloze:Text}}<br><br><small>{{Extra}}</small><br><small>{{Source}}</small>",
			height=120,
		)
		model_css = st.text_area(
			"Model CSS",
			value="""
.card {
	font-family: Georgia, serif;
	font-size: 20px;
	text-align: left;
	color: #1f2937;
	background-color: #f8fafc;
	line-height: 1.35;
	padding: 20px;
}
hr#answer { border: 0; border-top: 1px solid #d1d5db; margin: 14px 0; }
""".strip(),
			height=170,
		)

anki_config = {
	"deck_name_override": deck_name_override,
	"model_name": model_name_override,
	"deck_description": deck_description,
	"global_tags": global_tags,
	"include_reverse": include_reverse,
	"include_extra": include_extra,
	"include_source": include_source,
	"card_instructions": card_instructions,
	"basic_qfmt": basic_qfmt,
	"basic_afmt": basic_afmt,
	"reverse_qfmt": reverse_qfmt,
	"reverse_afmt": reverse_afmt,
	"cloze_qfmt": cloze_qfmt,
	"cloze_afmt": cloze_afmt,
	"model_css": model_css,
}

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

col1, col2 = st.columns([1, 1])
with col1:
	process_clicked = st.button("1) Process PDF", use_container_width=True)
with col2:
	generate_clicked = st.button("2) Generate Flashcards", use_container_width=True)

if process_clicked:
	if not uploaded_pdf:
		st.error("Please upload a PDF first.")
	else:
		with st.spinner("Loading and chunking PDF..."):
			pages = load_pdf_pages(uploaded_pdf)
			chunks = split_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

			embeddings = get_embeddings("BAAI/bge-m3")
			vectorstore = FAISS.from_documents(chunks, embeddings)

			st.session_state["pages"] = pages
			st.session_state["chunks"] = chunks
			st.session_state["vectorstore"] = vectorstore
			st.session_state["pdf_name"] = Path(uploaded_pdf.name).stem

		st.success("PDF processed successfully.")
		st.info(f"Pages: {len(pages)} | Chunks: {len(chunks)}")

if "vectorstore" in st.session_state:
	st.subheader("Flashcard Generation")
	gen_col1, gen_col2, gen_col3, gen_col4 = st.columns(4)
	with gen_col1:
		max_chunks = st.number_input("Max chunks to use", min_value=5, max_value=120, value=24, step=1)
	with gen_col2:
		cards_per_chunk = st.number_input(
			"Target cards per chunk", min_value=1, max_value=6, value=2, step=1
		)
	with gen_col3:
		context_k = st.number_input("Retriever context k", min_value=2, max_value=10, value=4, step=1)
	with gen_col4:
		note_type = st.selectbox("Note type", ["Basic", "Basic + Reverse", "Cloze"], index=0)

	if note_type == "Basic":
		card_type = "Basic Q/A"
	elif note_type == "Basic + Reverse":
		card_type = "Basic Q/A"
	else:
		card_type = "Cloze Deletion"

	if generate_clicked:
		if not groq_api_key:
			st.error("Provide GROQ_API_KEY in the sidebar or .env file.")
		else:
			with st.spinner("Generating cards with retrieval support..."):
				llm = ChatGroq(
					groq_api_key=groq_api_key,
					model_name=model_name,
					temperature=0,
				)

				vectorstore = st.session_state["vectorstore"]
				selected_chunks = choose_representative_chunks(vectorstore, max_chunks=max_chunks)

				all_cards = []
				progress = st.progress(0)

				for idx, chunk in enumerate(selected_chunks, start=1):
					query_for_support = chunk.page_content[:300]
					support_docs = vectorstore.similarity_search(query_for_support, k=int(context_k))
					support_text = "\n\n".join(d.page_content for d in support_docs)

					source_page = int(chunk.metadata.get("page", -1)) + 1
					cards = generate_cards_for_chunk(
						llm=llm,
						chunk_text=chunk.page_content,
						supporting_context=support_text,
						cards_per_chunk=int(cards_per_chunk),
						source_page=source_page,
						card_type=card_type,
						card_instructions=anki_config["card_instructions"],
					)
					all_cards.extend(cards)
					progress.progress(idx / len(selected_chunks))

				deduped_cards = deduplicate_cards(all_cards)
				st.session_state["cards"] = deduped_cards
				st.session_state["card_type"] = card_type
				st.session_state["note_type"] = note_type
				st.session_state["anki_config"] = {
					**anki_config,
					"include_reverse": anki_config["include_reverse"] or note_type == "Basic + Reverse",
				}

			st.success(f"Generated {len(deduped_cards)} unique cards.")

if "cards" in st.session_state and st.session_state["cards"]:
	cards = st.session_state["cards"]
	st.subheader("Preview")
	st.dataframe(cards[:20], use_container_width=True)

	deck_base = st.session_state.get("pdf_name", "RAG_Deck")
	stored_anki_config = st.session_state.get("anki_config", anki_config)
	timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
	selected_card_type = st.session_state.get("card_type", "Basic Q/A")
	if stored_anki_config.get("deck_name_override", "").strip():
		deck_base = stored_anki_config["deck_name_override"].strip()
	deck_name = f"{deck_base}::{selected_card_type}::{timestamp}"

	csv_bytes = cards_to_csv_bytes(cards)
	apkg_bytes = cards_to_apkg_bytes(
		cards,
		deck_name=deck_name,
		card_type=selected_card_type,
		anki_config=stored_anki_config,
	)

	dl_col1, dl_col2 = st.columns(2)
	with dl_col1:
		st.download_button(
			label="Download CSV",
			data=csv_bytes,
			file_name=f"{deck_base}_flashcards.csv",
			mime="text/csv",
			use_container_width=True,
		)
	with dl_col2:
		st.download_button(
			label="Download Anki Package (.apkg)",
			data=apkg_bytes,
			file_name=f"{deck_base}_{selected_card_type.lower().replace(' ', '_')}_flashcards.apkg",
			mime="application/octet-stream",
			use_container_width=True,
		)

	with st.expander("First 5 cards"):
		for i, c in enumerate(cards[:5], start=1):
			if c.get("card_type") == "Cloze Deletion":
				st.markdown(f"**{i}. Cloze**")
				st.write(c.get("text", ""))
			else:
				st.markdown(f"**{i}. {c['question']}**")
				st.write(c["answer"])
			if c["extra"]:
				st.caption(c["extra"])
