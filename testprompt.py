import streamlit as st
import openai
from openai import AzureOpenAI
import json
import base64
import os
import pandas as pd
import tempfile
import html
import traceback
import re
import docx  # Added for Word document processing
from dotenv import load_dotenv

# Azure Search imports
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery  # For Hybrid Search

# Langchain imports
from langchain_openai import AzureChatOpenAI as LangchainAzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import HumanMessage

load_dotenv()
# >>> st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND <<<
try:
    st.set_page_config(
        page_title=" BIAL Multi-Agent Regulatory Platform", page_icon="✈️", layout="wide"
    )
except Exception as e_config:
    print(f"CRITICAL ERROR during st.set_page_config: {e_config}")
    st.error(f"Error during st.set_page_config: {e_config}")
    st.stop()


# --- Helper Function ---
def check_creds(cred_value, placeholder_prefix="YOUR_"):
    """Checks if a credential value is missing or looks like a placeholder."""
    if not cred_value:
        return True
    if isinstance(cred_value, str):
        if placeholder_prefix in cred_value.upper():
            return True
        if "ENTER_YOUR" in cred_value.upper():
            return True
        if cred_value.startswith("<") and cred_value.endswith(">"):
            return True
    return False


def main_app_logic():
    ###################################
    # Azure Cognitive Search Credentials --- REPLACE THESE ---
    ###################################
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
    DEFAULT_AZURE_SEARCH_INDEX_NAME = os.getenv("DEFAULT_AZURE_SEARCH_INDEX_NAME")
    DEFAULT_VECTOR_FIELD_NAME = os.getenv("DEFAULT_VECTOR_FIELD_NAME")
    DEFAULT_SEMANTIC_CONFIG_NAME = os.getenv("DEFAULT_SEMANTIC_CONFIG_NAME")

    ###################################
    # Azure OpenAI Credentials --- REPLACE THESE ---
    ###################################
    AZURE_OPENAI_ENDPOINT_VAL = os.getenv("AZURE_OPENAI_ENDPOINT_VAL")
    AZURE_OPENAI_API_VERSION_VAL = os.getenv("AZURE_OPENAI_API_VERSION_VAL")
    AZURE_OPENAI_API_KEY_VAL = os.getenv("AZURE_OPENAI_API_KEY_VAL")

    DEPLOYMENT_ID_VAL = os.getenv("DEPLOYMENT_ID_VAL")
    PLANNING_LLM_DEPLOYMENT_ID = os.getenv("PLANNING_LLM_DEPLOYMENT_ID")
    SEARCH_QUERY_EMBEDDING_DEPLOYMENT_ID = os.getenv(
        "SEARCH_QUERY_EMBEDDING_DEPLOYMENT_ID"
    )

    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT_VAL
    openai.api_version = AZURE_OPENAI_API_VERSION_VAL
    openai.api_key = AZURE_OPENAI_API_KEY_VAL

    search_query_embeddings_model = None
    planning_openai_client = None  # Client for planning LLM
    synthesis_openai_client = None  # Client for synthesis LLM

    try:
        if not (
            check_creds(AZURE_OPENAI_API_KEY_VAL)
            or check_creds(AZURE_OPENAI_ENDPOINT_VAL)
            or check_creds(SEARCH_QUERY_EMBEDDING_DEPLOYMENT_ID)
            or check_creds(AZURE_OPENAI_API_VERSION_VAL)
        ):
            search_query_embeddings_model = AzureOpenAIEmbeddings(
                azure_deployment=SEARCH_QUERY_EMBEDDING_DEPLOYMENT_ID,
                azure_endpoint=AZURE_OPENAI_ENDPOINT_VAL,
                api_key=AZURE_OPENAI_API_KEY_VAL,
                api_version=AZURE_OPENAI_API_VERSION_VAL,
                chunk_size=1,
            )
        else:
            st.sidebar.warning(
                f"Search Query Embeddings: Credentials for '{SEARCH_QUERY_EMBEDDING_DEPLOYMENT_ID}' appear to be placeholders.",
                icon="⚠️",
            )
    except Exception as e:
        st.sidebar.error(
            f"Error initializing Search Query Embeddings ('{SEARCH_QUERY_EMBEDDING_DEPLOYMENT_ID}'): {e}"
        )

    try:
        if not (
            check_creds(AZURE_OPENAI_API_KEY_VAL)
            or check_creds(AZURE_OPENAI_ENDPOINT_VAL)
            or check_creds(AZURE_OPENAI_API_VERSION_VAL)
        ):
            if not check_creds(PLANNING_LLM_DEPLOYMENT_ID):
                planning_openai_client = AzureOpenAI(
                    api_key=AZURE_OPENAI_API_KEY_VAL,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT_VAL,
                    api_version=AZURE_OPENAI_API_VERSION_VAL,
                )
            else:
                st.sidebar.warning(
                    f"Planning LLM: Deployment ID '{PLANNING_LLM_DEPLOYMENT_ID}' is a placeholder.",
                    icon="⚠️",
                )

            if not check_creds(DEPLOYMENT_ID_VAL):
                synthesis_openai_client = AzureOpenAI(
                    api_key=AZURE_OPENAI_API_KEY_VAL,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT_VAL,
                    api_version=AZURE_OPENAI_API_VERSION_VAL,
                )
            else:
                st.sidebar.warning(
                    f"Synthesis LLM: Deployment ID '{DEPLOYMENT_ID_VAL}' is a placeholder.",
                    icon="⚠️",
                )
        else:
            st.sidebar.warning(
                "Main OpenAI Client: Credentials appear to be placeholders.", icon="⚠️"
            )
    except Exception as e:
        st.sidebar.error(f"Error initializing AzureOpenAI clients: {e}")

    def get_query_vector(text_to_embed):
        if not search_query_embeddings_model:
            st.toast(
                "Search Query Embedding model not ready. Hybrid search might be affected.",
                icon="⚠️",
            )
            return None
        try:
            return search_query_embeddings_model.embed_query(text_to_embed)
        except Exception as e:
            st.error(
                f"Error generating query vector for '{text_to_embed[:50]}...': {e}"
            )
            print(f"Detailed error generating query vector: {traceback.format_exc()}")
            return None

    @st.cache_data(ttl=3600)
    def get_indexes():
        indexes = [
            (
                DEFAULT_AZURE_SEARCH_INDEX_NAME
                if DEFAULT_AZURE_SEARCH_INDEX_NAME
                else "default-index-placeholder"
            )
        ]
        if check_creds(AZURE_SEARCH_API_KEY) or check_creds(AZURE_SEARCH_ENDPOINT):
            st.sidebar.warning(
                "Azure Search: Credentials appear to be placeholders. Using default index name.",
                icon="⚠️",
            )
            return list(set(indexes))
        try:
            search_creds = AzureKeyCredential(AZURE_SEARCH_API_KEY)
            index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, search_creds)
            fetched_indexes = [index.name for index in index_client.list_indexes()]
            if fetched_indexes:
                indexes.extend(fetched_indexes)
            else:
                st.sidebar.info(
                    f"No additional indexes found via API. Check search service.",
                    icon="ℹ️",
                )
        except Exception as e:
            st.sidebar.error(
                f"Cannot retrieve Azure Search indexes: {e}. Check credentials and endpoint.",
                icon="🚨",
            )
        return list(set(indexes))

    def query_azure_search(
        query_text,
        index_name,
        k=5,
        use_hybrid_semantic_search=True,
        vector_field_name=DEFAULT_VECTOR_FIELD_NAME,
        semantic_config_name=DEFAULT_SEMANTIC_CONFIG_NAME,
    ):
        context = ""
        references_data = []
        if (
            check_creds(AZURE_SEARCH_API_KEY)
            or check_creds(AZURE_SEARCH_ENDPOINT)
            or check_creds(index_name)
        ):
            return (
                "Error: Azure Search credentials or index name appear to be placeholders.",
                [],
            )
        try:
            search_client = SearchClient(
                AZURE_SEARCH_ENDPOINT,
                index_name,
                AzureKeyCredential(AZURE_SEARCH_API_KEY),
            )
            select_fields = ["content", "filepath", "url", "title"]
            search_kwargs = {
                "search_text": query_text if query_text and query_text.strip() else "*",
                "top": k,
                "include_total_count": True,
                "select": ",".join(select_fields),
            }
            if use_hybrid_semantic_search:
                if vector_field_name and not check_creds(vector_field_name):
                    query_vector = get_query_vector(query_text)
                    if query_vector:
                        vector_q = VectorizedQuery(
                            vector=query_vector,
                            k_nearest_neighbors=k,
                            fields=vector_field_name,
                        )
                        search_kwargs["vector_queries"] = [vector_q]
                    else:
                        st.toast(
                            "Warning: Could not generate query vector for hybrid search.",
                            icon="⚠️",
                        )
                elif vector_field_name:
                    st.toast(
                        f"Warning: Vector field name ('{vector_field_name}') is a placeholder. Vectors not used.",
                        icon="⚠️",
                    )

                if semantic_config_name and not check_creds(semantic_config_name):
                    search_kwargs.update(
                        {
                            "query_type": "semantic",
                            "semantic_configuration_name": semantic_config_name,
                            "query_caption": "extractive",
                            "query_answer": "extractive",
                        }
                    )
                elif semantic_config_name:
                    st.toast(
                        f"Warning: Semantic config name ('{semantic_config_name}') is a placeholder. Semantic ranking skipped.",
                        icon="⚠️",
                    )

            results = search_client.search(**search_kwargs)

            if results.get_count() == 0:
                return "", []
            processed_references = {}
            for idx, doc in enumerate(results):
                context += doc.get("content", "") + "\n\n"
                doc_title = doc.get("title")
                doc_filepath = doc.get("filepath")
                doc_url = doc.get("url")
                display_name = (
                    doc_title
                    if doc_title
                    else (
                        os.path.basename(doc_filepath)
                        if doc_filepath
                        else f"Source {idx+1}"
                    )
                )
                ref_key = doc_url or display_name
                content_snippet_to_store = ""
                if doc.get("@search.captions"):
                    caption = doc["@search.captions"][0]
                    content_snippet_to_store = (
                        caption.highlights
                        if caption.highlights and caption.highlights.strip()
                        else caption.text
                    )
                else:
                    content_snippet_to_store = doc.get("content", "")[:250] + "..."
                if ref_key not in processed_references:
                    processed_references[ref_key] = {
                        "filename_or_title": display_name,
                        "url": doc_url,
                        "score": doc.get("@search.score"),
                        "reranker_score": doc.get("@search.reranker_score"),
                        "content_snippet": content_snippet_to_store,
                    }
            references_data = list(processed_references.values())
        except Exception as e:
            return f"Error accessing search index '{index_name}'. Details: {str(e)}", []
        return context.strip(), references_data

    def get_query_plan_from_llm(user_original_question, client_for_planning):
        if not client_for_planning:
            return (
                "Error: Planning LLM client not initialized. Check Azure OpenAI credentials and deployment.",
                None,
            )
        if check_creds(PLANNING_LLM_DEPLOYMENT_ID):
            return (
                f"Error: Planning LLM Deployment ID '{PLANNING_LLM_DEPLOYMENT_ID}' is a placeholder.",
                None,
            )

        planning_prompt = f"""You are a query planning assistant specializing in breaking down complex questions about **AERA regulatory documents, often concerning tariff orders, consultation papers, control periods, and specific financial data (like CAPEX, Opex, Traffic) for airport operators such as DIAL, MIAL, BIAL, HIAL.**

Your primary task is to take a user's complex question related to these topics and break it down into a series of 1 to 20 simple, self-contained search queries that can be individually executed against a document index. Each search query should aim to find a specific piece of information (e.g., a specific figure, a justification, a comparison point) needed to answer the overall complex question.

If the user's question is already simple and can be answered with a single search, return just that single query in the list.
If the question is very complex and might require more distinct search steps, formulate the most critical 1 to 20 search queries, focusing on distinct pieces of information.

Return your response ONLY as a JSON list of strings, where each string is a search query.
For example, if the user asks: "Compare the approved CAPEX for DIAL and MIAL for the fourth control period, and also find the main justifications provided by AERA for any differences."
You might return:
[
  "AERA approved CAPEX for DIAL fourth control period",
  "AERA approved CAPEX for MIAL fourth control period",
  "AERA justifications for CAPEX differences between DIAL and MIAL for fourth control period"
]

User's complex question: {user_original_question}

Your JSON list of search queries:
"""
        try:
            response = client_for_planning.chat.completions.create(
                model=PLANNING_LLM_DEPLOYMENT_ID,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.0,
                max_tokens=16000,
            )
            plan_str = response.choices[0].message.content
            try:
                # The LLM can return JSON wrapped in markdown ```json ... ``` or with other text.
                # This regex extracts the first valid JSON array from the string.
                match = re.search(r"\[.*\]", plan_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    query_plan = json.loads(json_str)
                else:
                    # If no array is found, assume the whole string is the JSON and try to parse
                    query_plan = json.loads(plan_str)

                if isinstance(query_plan, list) and all(
                    isinstance(q, str) for q in query_plan
                ):
                    return None, query_plan
                else:
                    # This case handles if the JSON is valid but not a list of strings
                    st.warning(
                        f"LLM plan was valid JSON but not a list of strings: {plan_str}. Using original query."
                    )
                    return None, [user_original_question]
            except json.JSONDecodeError:
                # This case handles if no valid JSON could be parsed at all
                st.warning(
                    f"LLM plan was not valid JSON: {plan_str}. Using original query."
                )
                return None, [user_original_question]
        except Exception as e:
            return f"Error getting query plan from LLM: {e}", None

    def generate_answer_from_search(
        user_question,
        index_name,
        use_hybrid_semantic,
        vector_field,
        semantic_config,
        temperature,
        max_tokens_param,
        client_for_synthesis,
        show_details=True,
    ):
        if not client_for_synthesis:
            return "Error: Synthesis LLM client not initialized. Check Azure OpenAI credentials and deployment."
        if check_creds(DEPLOYMENT_ID_VAL):
            return f"Config Error: Synthesis LLM Deployment ID '{DEPLOYMENT_ID_VAL}' is a placeholder."
        if (
            check_creds(AZURE_OPENAI_API_KEY_VAL)
            or check_creds(AZURE_OPENAI_ENDPOINT_VAL)
            or check_creds(AZURE_OPENAI_API_VERSION_VAL)
        ):
            return (
                "Config Error: Search Q&A OpenAI credentials appear to be placeholders."
            )
        if use_hybrid_semantic and (
            check_creds(vector_field) or not search_query_embeddings_model
        ):
            return "Config Error: Hybrid search selected, but vector field name is a placeholder or embedding model is not initialized."

        if show_details:
            st.write("⚙️ Generating query plan...")
        plan_error, query_plan = get_query_plan_from_llm(
            user_question, planning_openai_client
        )

        if show_details:
            if plan_error:
                st.error(plan_error)
                query_plan = [user_question]
                st.write(
                    "⚠️ Planning failed, proceeding with the original question as a single search query."
                )
            if not query_plan:
                query_plan = [user_question]
                st.write(
                    "⚠️ Query plan was empty, proceeding with the original question."
                )

            st.write(f"📝 **Execution Plan:**")
            for i, q_step in enumerate(query_plan):
                st.write(f"    Step {i+1}: {q_step}")

        all_contexts = []
        all_retrieved_details = []
        combined_context_for_llm = ""

        for i, sub_query in enumerate(query_plan):
            if show_details:
                st.write(f"🔍 Executing Step {i+1}: Searching for '{sub_query}'...")
            context_for_step, retrieved_details_for_step = query_azure_search(
                query_text=sub_query,
                index_name=index_name,
                k=5,
                use_hybrid_semantic_search=use_hybrid_semantic,
                vector_field_name=vector_field,
                semantic_config_name=semantic_config,
            )
            if isinstance(context_for_step, str) and context_for_step.startswith(
                "Error"
            ):
                if show_details:
                    st.warning(
                        f"    ⚠️ Error in sub-query '{sub_query}': {context_for_step}"
                    )
                continue

            if context_for_step:
                combined_context_for_llm += (
                    f"\n\n--- Context for sub-query: '{sub_query}' ---\n"
                    + context_for_step
                )
                all_retrieved_details.extend(retrieved_details_for_step)
            elif show_details:
                st.write(f"    ℹ️ No results found for sub-query: '{sub_query}'")

        if show_details:
            st.session_state.last_retrieved_chunks_search = all_retrieved_details

        if not combined_context_for_llm.strip():
            return "No relevant information found in search index across all planned queries."

        if show_details:
            st.write("💡 Synthesizing final answer...")
        formatted_sources = []
        if all_retrieved_details:
            unique_sources_dict = {}
            for item in all_retrieved_details:
                key = item.get("url") or item.get("filename_or_title")
                if key not in unique_sources_dict:
                    unique_sources_dict[key] = item

            for i, item in enumerate(list(unique_sources_dict.values())):
                source_line = f"Source {i+1}: {html.escape(item['filename_or_title'])}"
                if item.get("url"):
                    source_line += f" ([Link]({html.escape(item['url'])}))"
                if item.get("score") is not None:
                    source_line += f" [Score: {item['score']:.4f}]"
                if item.get("reranker_score") is not None:
                    source_line += f" [Reranker Score: {item['reranker_score']:.4f}]"
                formatted_sources.append(source_line)
        else:
            formatted_sources.append(
                "No specific source metadata retrieved, but context was found."
            )
        formatted_refs_str = "\n".join(formatted_sources)

        user_base_prompt_instructions = """Part 1:   "you are AI assistant for Multiyear tarrif submission for AERA. Answer concisely based only on the provided CONTEXT. If the CONTEXT doesn't contain the answer, state that clearly. Final reponse must have 1500 words at least.

* **3.2. Source Attribution (Authority Data - Handling Terminology for Authority's Stance):**
    If the query relates to the "Authority's" stance (e.g., user asks “What is the authority’s *approved* change?”, “What did the authority *decide*?”, “What was *proposed* by the authority?”), your primary goal is to find the authority's documented action or position on that specific subject *within the CONTEXT*.

    * **Understanding Levels of Finality (for AI's internal logic):**
        * **Final/Conclusive Terms:** "approved by the authority", "decided by the authority", "authority's decision", "final tariff/order states", "sanctioned by authority", "adopted by authority".
        * **Provisional/Draft Terms:** "proposed by theauthority", "authority's proposal", "draft figures", "recommended by the authority" (if it's a recommendation for a later decision).
        * **Analytical/Consideration Terms:** "considered by the authority", "analyzed by the authority", "examined by the authority", "authority's review/view/assessment/preliminary findings".

    * **Extraction Strategy Based on User Intent and Document Content:**
        1.  **Attempt to Match User's Exact Term First:** Always search the CONTEXT for information explicitly matching the user's specific terminology (e.g., if the user asks for "approved," look first for "approved by the authority"). If found, present this.
        2.  **If User's Query Implies Finality (e.g., asks for "approved," "final figures," "decision"):**
            * And their *exact term* is NOT found in the CONTEXT for that item:
                * **Prioritize searching the CONTEXT for other Final/Conclusive terms** (e.g., "decided by the authority," "authority's decision"). If one of these is found, present this information. You MUST then state clearly: "You asked for 'approved.' The document describes what was '*[actual term found, e.g., decided by the authority]*' as follows: \[data and references]."
                * If no Final/Conclusive terms are found for that item in the CONTEXT, then (and only then) look for Provisional/Draft terms (e.g., "proposed by the authority"). If found, present this, stating: "You asked for 'approved.' A final approval or decision was not found for this item in the provided context. However, the authority '*proposed*' the following: \[data and references]."
                * If neither of the above is found, look for Analytical/Consideration terms and report similarly with clarification.
        3.  **If User's Query Uses a Provisional/Draft Term (e.g., "proposed"):**
            * Prioritize finding information matching those Provisional/Draft terms in the CONTEXT.
            * If not found, you can then look for Analytical/Consideration terms, clarifying the terminology. Avoid presenting Final/Conclusive terms unless you explicitly state that the user asked for a draft but a final version was found regarding that specific point.
        4.  **If User's Query Uses an Analytical/Consideration Term (e.g., "considered"):**
            * Prioritize finding information matching those terms. Always give me the table refrence for the response i mean which table no you have refered for the response

    * **Accurate Reporting is Key:** Always present information using the **document's actual terminology**. Clearly explain if and how it relates to the user's original query terms. If multiple relevant stages of authority action are evident in the CONTEXT (e.g., a proposal and then a later decision), you may summarize both, clearly distinguishing them by the terms used in the document.
    * **If No Relevant Authority Action Found:** If the provided CONTEXT contains no clear information matching any relevant stage of authority action (final, provisional, or analytical) regarding the specific subject of the query, state that this information was not found for the authority in the provided context.
    * **Table Headers:** Use data from tables if their headers clearly indicate the source and nature of the data (e.g., "Figures as Decided by the Authority," "Operator's Proposed Traffic").
"**Crucial Instruction for Authority's Stance:** When the user asks for 'approved', 'final', or 'decided' figures from the Authority, it is **imperative** that you prioritize extracting text explicitly labeled with terms like 'decided by the authority' or 'approved by the authority' from the CONTEXT. If such conclusive terms are present for the queried item, present them. Only if NO such conclusive terms are found in the CONTEXT for that item should you then present information labeled 'proposed by the authority', and you MUST clearly state that you are providing 'proposed' figures because 'approved/decided' ones were not found in the given context."
* **Clarifying "Considered by Authority" in a "True-Up" Context:**
    * If the user query asks what the authority 'considered' in relation to a 'true-up' of a specific control period:
        * First, check if the CONTEXT contains information about the authority's **analysis or verification of the actual figures submitted for that true-up period**. If so, present this.
        * If the user's query might also imply understanding the **original baseline** that is being trued-up against, you can additionally (or if the above is not found) look for what the authority **originally considered or determined when setting the tariffs for that control period at the beginning of that period**.
        * **Crucially, always differentiate these two.** For example: "For the true-up of the Third Control Period, DIAL submitted the following actual traffic figures (e.g., from Table 25): [data]. The figures that the Authority had originally considered at the time of determining the tariff for the Third Control Period were (e.g., from Table 26): [data]."
        * If the query is simply "What was considered for true-up..." without specifying "original determination" vs "actuals review", and both types of information are in the context, you might offer both or ask the user to clarify which aspect of "considered for true-up" they are interested in Mandatory Table Referencing:
For any data, figures, or claims extracted from a table within the CONTEXT, you must cite the corresponding table number in your response. This is a strict requirement for all outputs. The reference should be placed directly with the data it pertains to.

Example 1: "The Authority approved Aeronautical Revenue of ₹1,500 Cr for FY 2024-25 (Table 15)."
Example 2: "For the true-up, the operator submitted actual passenger traffic of 45 million (as per Table 3.2), while the original figure considered by the Authority was 42 million (from Table 5.1 of the original Order)."""
        synthesis_prompt = (
            f"{user_base_prompt_instructions}\n\n"
            f"Based on the general background above, please synthesize a comprehensive answer to the original USER QUESTION using all the following retrieved CONTEXT from multiple search steps and the IDENTIFIED CONTEXT SOURCES.\n\n"
            f"ORIGINAL USER QUESTION: {user_question}\n\n"
            f"AGGREGATED CONTEXT (from multiple search steps):\n---------------------\n{combined_context_for_llm}\n---------------------\n\n"
            f"IDENTIFIED CONTEXT SOURCES (from metadata):\n---------------------\n{formatted_refs_str}\n---------------------\n\n"
            f"SPECIFIC INSTRUCTIONS FOR YOUR RESPONSE (in addition to the general background provided):\n"
            f"1. Directly address all parts of the ORIGINAL USER QUESTION.\n"
            f"2. Synthesize information from the different context sections if they relate to different aspects of the original question.\n"
            f"3. Format numerical data extracted from tables into an HTML table with borders (e.g., <table border='1'>...). Use table headers (<th>) and table data cells (<td>).\n"
            f"4. **References are crucial.** At the end of your answer, include a 'References:' section listing the source documents (using filenames or titles as provided in 'IDENTIFIED CONTEXT SOURCES') from which the information was derived. If a URL is available for a source, make the filename/title a clickable hyperlink to that URL.\n\n"
            f"COMPREHENSIVE ANSWER TO THE ORIGINAL USER QUESTION:\n"
        )
        try:
            response = client_for_synthesis.chat.completions.create(
                model=DEPLOYMENT_ID_VAL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=temperature,
                max_tokens=max_tokens_param,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error OpenAI (Synthesis): {e}")
            return f"Error generating synthesized answer: {e}"

    def extract_text_from_docx(file):
        """Extracts text from a .docx file."""
        try:
            document = docx.Document(file)
            return "\n".join([para.text for para in document.paragraphs])
        except Exception as e:
            st.error(f"Error reading Word document: {e}")
            return None

    @st.cache_data
    def get_base64_image(image_path: str) -> str | None:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8")
        except FileNotFoundError:
            print(f"Warning: Logo not found at {image_path}")
            return None
        except Exception as e:
            print(f"Warning: Logo load error for {image_path}: {e}")
            return None

    main_logo_path = "bial_logo.png"
    main_logo_base64 = get_base64_image(main_logo_path)

    st.markdown(
        """ <style>
    :root {
        --primary-background: #33263C; /* Dominant dark purple/brown background */
        --card-background: #2A2A2A; /* Dark gray of the card/table header */
        --text-color-light: #E0E0E0; /* Light gray for most text */
        --text-color-dark: #A0A0A0; /* Slightly darker gray for some text or secondary information */
        --header-text-color: #FFFFFF; /* White for the header "Deploy" and column titles */
        --button-green: #28A745; /* Green color of the "Generate" button */
        --sidebar-dark: #1E1E1E; /* Very dark almost black for the sidebar area */
    }

    /* General Styles */
    body {
        color: var(--text-color-light);
    }
    .stApp {
        background-color: var(--primary-background);
    }

    /* Main content area (response box, text area, etc.) */
    .response-box, .stTextArea, .stTextInput, .st-expander {
        background-color: var(--card-background) !important;
        border: 1px solid #4A4A4A !important; /* A slightly lighter border for definition */
        color: var(--text-color-light) !important;
        border-radius: 8px;
    }
    .st-expander header {
        color: var(--text-color-light) !important;
    }

    .response-box {
        padding: 20px;
        margin-bottom: 20px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* History Entry in Sidebar */
    .history-entry {
        border: 1px solid #4A4A4A;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        background-color: var(--card-background);
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    /* Titles and Headers */
    .title-bar h1, h1, h2, h3, h4, p {
        color: var(--header-text-color);
    }
    
    /* --- CSS FOR ALL BUTTONS TO BE GREEN --- */

    /* Style for all buttons */
    .stButton>button {
        border: none;
        background-color: var(--button-green);
        color: var(--header-text-color);
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #218838; /* A darker green for hover */
        box-shadow: 0 0 15px rgba(40, 167, 69, 0.5);
    }
    
    .stButton>button:active {
        background-color: #1E7E34 !important; /* An even darker green for active/click */
    }

    /* Specific layout tweaks for the predefined question buttons */
    div[data-testid="column"] .stButton>button {
        height: 100%;
        width: 100%;
        text-align: left !important;
        padding: 15px;
        font-weight: normal;
    }

    div[data-testid="column"] .stButton>button:hover {
        transform: translateY(-3px);
    }
    
    /* --- END OF BUTTON CSS --- */

    /* Sidebar and Main Content Area Backgrounds */
    .st-emotion-cache-1r6slb0 { /* Sidebar */
        background-color: var(--sidebar-dark);
    }
    .st-emotion-cache-16txtl3 { /* Main content area */
        background-color: var(--primary-background);
    }
    
    </style>
    
    
    """,
        unsafe_allow_html=True,
    )

    default_session_state = {
        "conversation_history": [],
        "last_retrieved_chunks_search": [],
        "question_text": "",
        "conv_agent_temp": 0.5,
        "conv_agent_max_tokens": 16000,
        "conv_agent_selected_index": DEFAULT_AZURE_SEARCH_INDEX_NAME,
        "conv_agent_use_hybrid": True,
        "conv_agent_vector_field": DEFAULT_VECTOR_FIELD_NAME,
        "conv_agent_semantic_config": DEFAULT_SEMANTIC_CONFIG_NAME,
        "app_mode": "Conversational Agent",  # Default app mode
        "mda_analysis_type": "MDA Manpower Analysis",  # Default analysis type for MDA Reviewer
    }
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # =================================================================================
    # START OF MODIFIED SIDEBAR CODE
    # =================================================================================
    with st.sidebar:
        st.sidebar.title("Choose App")
        st.session_state.app_mode = st.sidebar.radio(
            "Select the application to use:",
            ("Conversational Agent", "MDA Reviewer"),
            key="app_mode_selector",
        )
        st.markdown("---")

        with st.expander("⚙️ Settings & History", expanded=True):
            st.header("Agent Settings")
            st.subheader("Response Setting")
            st.session_state.conv_agent_temp = st.slider(
                "Temperature",
                0.0,
                1.0,
                st.session_state.conv_agent_temp,
                0.1,
                key="temp_search_conv",
            )
            st.session_state.conv_agent_max_tokens = st.slider(
                "Output",
                100,
                16000,
                st.session_state.conv_agent_max_tokens,
                50,
                key="max_tokens_search_conv",
            )
            st.markdown("---")
            st.subheader("Regulatory database")
            available_indexes = get_indexes()
            try:
                default_idx_pos = (
                    available_indexes.index(st.session_state.conv_agent_selected_index)
                    if st.session_state.conv_agent_selected_index in available_indexes
                    else 0
                )
            except ValueError:
                default_idx_pos = 0
            st.session_state.conv_agent_selected_index = st.selectbox(
                "Select Index",
                available_indexes,
                index=default_idx_pos,
                key="index_selector_conv",
            )
            if not st.session_state.conv_agent_selected_index and available_indexes:
                st.session_state.conv_agent_selected_index = available_indexes[0]
            st.session_state.conv_agent_use_hybrid = st.checkbox(
                "Enable Multi-Agent Flow",
                value=st.session_state.conv_agent_use_hybrid,
                key="hybrid_search_checkbox_conv",
            )
            if st.session_state.conv_agent_use_hybrid:
                st.session_state.conv_agent_vector_field = st.text_input(
                    "Vector Field Name in Index",
                    value=st.session_state.conv_agent_vector_field,
                    key="vector_field_conv",
                    help="e.g., content_vector",
                )
                st.session_state.conv_agent_semantic_config = st.text_input(
                    "Semantic Configuration Name",
                    value=st.session_state.conv_agent_semantic_config,
                    key="semantic_config_conv",
                    help="e.g., azureml-default",
                )

            if st.session_state.app_mode == "Conversational Agent":
                st.markdown("---")
                if st.session_state.conversation_history:
                    st.subheader("📜 Chat History")
                    for i, entry in enumerate(
                        reversed(st.session_state.conversation_history)
                    ):
                        st.markdown(
                            f"**Q:** {entry['question']}", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div class='history-entry'>{entry['answer']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("---", unsafe_allow_html=True)

                    if st.button("Clear History"):
                        st.session_state.conversation_history = []
                        st.rerun()
    # =================================================================================
    # END OF MODIFIED SIDEBAR CODE
    # =================================================================================

    col_header1, col_header2 = st.columns([1, 10])
    with col_header1:
        if main_logo_base64:
            st.image(f"data:image/png;base64,{main_logo_base64}", width=400)
    with col_header2:
        st.markdown(
            '<div class="title-bar"><h1> BIAL Regulatory Assistant </h1></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='margin-top: 0; margin-bottom:1em;'>", unsafe_allow_html=True
    )

    # --- App selection logic ---
    if st.session_state.app_mode == "Conversational Agent":
        # --- Conversational Agent UI ---
        st.subheader(
            f"💬 Conversational Agent (Index: {st.session_state.conv_agent_selected_index or 'Not Selected'})"
        )
        if st.session_state.last_retrieved_chunks_search:
            with st.expander(
                "View Retrieved Context Sources (from last manual query)",
                expanded=False,
            ):
                if not st.session_state.last_retrieved_chunks_search:
                    st.write("No context sources were retrieved for the last query.")
                for i, item in enumerate(st.session_state.last_retrieved_chunks_search):
                    title = html.escape(item.get("filename_or_title", f"Source {i+1}"))
                    url = item.get("url")
                    snippet = html.escape(item.get("content_snippet", "N/A"))
                    score_info = ""
                    if item.get("score") is not None:
                        score_info += f" Score: {item['score']:.4f}"
                    if item.get("reranker_score") is not None:
                        score_info += f" Reranker: {item['reranker_score']:.4f}"
                    display_text = f"**{title}**{score_info}"
                    if url:
                        display_text += f" ([Link]({html.escape(url)}))"
                    st.markdown(display_text, unsafe_allow_html=True)
                    st.caption(f"Snippet: {snippet}")
                    st.markdown("---")

        st.markdown("---")
        st.subheader("Ask a question")
        input_label_main = "Your question about regulatory documents:"

        predefined_questions_search = [
            "Calculate and compare the YoY change of employee expenses of DIAL and MIAL for the fourth control period",
            "What is the YoY change of employee expenses submitted by MIAL for the fourth control period and the rationale for the growth rates",
            "What is the YoY change of employee expenses submitted by MIAL for the fourth control period and the rationale for the growth rates.",
            "Compare the manpower expense per total passenger traffic submitted by  DIAL and MIAL respectively for fourth control period.",
        ]

        # Display predefined questions as clickable boxes
        cols = st.columns(len(predefined_questions_search))
        for i, question in enumerate(predefined_questions_search):
            if cols[i].button(
                question, key=f"predef_q_{i}", help=question, use_container_width=True
            ):
                st.session_state.question_text = question

        def update_question_text():
            st.session_state.question_text = (
                st.session_state.question_text_area_main_common
            )

        st.text_area(
            input_label_main,
            value=st.session_state.question_text,
            key="question_text_area_main_common",
            height=100,
            on_change=update_question_text,
        )
        submit_button_main = st.button(
            "Submit Question", key="submit_q_button_main_common", type="primary"
        )

        if submit_button_main:
            user_query_main = st.session_state.question_text
            if not user_query_main.strip():
                st.warning("Please enter a question.", icon="⚠️")
            else:
                answer_main = None
                with st.spinner("Thinking..."):
                    if not st.session_state.conv_agent_selected_index or check_creds(
                        st.session_state.conv_agent_selected_index
                    ):
                        st.error(
                            "Conversational Agent: Valid Search Index not selected or is a placeholder.",
                            icon="🚨",
                        )
                    elif st.session_state.conv_agent_use_hybrid and (
                        check_creds(st.session_state.conv_agent_vector_field)
                        or not search_query_embeddings_model
                    ):
                        st.error(
                            "Conversational Agent: Hybrid search selected, but Vector Field Name is a placeholder or Embedding Model failed to initialize.",
                            icon="🚨",
                        )
                    elif not planning_openai_client or not synthesis_openai_client:
                        st.error(
                            "Conversational Agent: Core LLM clients for planning or synthesis not initialized. Check credentials.",
                            icon="�",
                        )
                    else:
                        answer_main = generate_answer_from_search(
                            user_query_main,
                            st.session_state.conv_agent_selected_index,
                            st.session_state.conv_agent_use_hybrid,
                            st.session_state.conv_agent_vector_field,
                            st.session_state.conv_agent_semantic_config,
                            st.session_state.conv_agent_temp,
                            st.session_state.conv_agent_max_tokens,
                            synthesis_openai_client,
                            show_details=True,
                        )

                if answer_main:
                    if isinstance(answer_main, str) and (
                        answer_main.startswith("Error:")
                        or answer_main.startswith("Config Error:")
                        or answer_main.startswith("Sorry,")
                    ):
                        st.error(answer_main, icon="🚨")
                    else:
                        st.subheader("💡 Answer")
                        st.markdown(
                            f'<div class="response-box">{answer_main}</div>',
                            unsafe_allow_html=True,
                        )
                        st.session_state.conversation_history.append(
                            {
                                "mode": "Conversational Agent",
                                "question": user_query_main,
                                "answer": str(answer_main),
                            }
                        )
                elif not (
                    not st.session_state.conv_agent_selected_index
                    or check_creds(st.session_state.conv_agent_selected_index)
                ):
                    st.info("No specific answer or error returned by the agent.")

    elif st.session_state.app_mode == "MDA Reviewer":
        # --- MYTP Reviewer UI ---
        st.subheader("📄Review and validation of MDA")
        st.info(
            "The tool analyses and validates the cases and justifications prepared by BIAL for regulatory submission using benchmarking of information from regulatory submissions of peer airports. Please upload the MDA document below.",
            icon="ℹ️",
        )

        # --- Dynamic Analysis Type Selection ---
        analysis_prompts_config = {
            "MDA Manpower Analysis": {
                "Analysis of manpower expenditure projection for BIAL for fourth control period": f"The uploaded document proposes the following: '{{document_summary}}'. Use the following steps for analysing the manpower expenditure projected by BIAL: 1- Year on Year growth of personnel cost projected by BIAL for fourth control period. 2- Justification for personnel cost growth in fourth control period provided by BIAL. 3- Year on Year growth of manpower expenses submitted by DIAL for fourth control period. 4- Justification provided by DIAL for manpower expenses submitted by DIAL for fourth control period. 5- Examination and rationale provided by authority for manpower expenses submitted by DIAL for fourth control period. 6- Year on Year growth of employee cost submitted by MIAL for fourth control period. 7- Justification provided by MIAL for manpower expenses submitted by MIAL for fourth control period. 8- Examination and rationale provided by authority for manpower expenses submitted by MIAL for fourth control period. 9- Using the rationale extracted in steps 4, 5 7 and 8 suggest how the rationale or justification provided by BIAL in the MDA document for manpower expenditure for fourth control period can be enhanced. For every suggestion made, give specific reason why the suggestion was made by you using relevant references from DIAL and MIAL tariff orders or consultation papers.",
                "Analysis of actual manpower expenditure for BIAL for third control period": f"The uploaded document proposes the following: '{{document_summary}}'. Use the following steps for analyzing the actual manpower expenditure for the third control period: 1. Actual manpower expenditure for BIAL and variance from authority approved manpower expenditure for the third control period. 2. Justification for manpower expenditure in third control period provided by BIAL. 3 Actual manpower expenditure for DIAL and variance from authority approved manpower expenditure for the third control period. 4. Justification provided by DIAL for actual manpower expenses for third control period and the reason for variance compared to authority approved figures. 5. Examination and rationale provided by authority for actual manpower expenditure for only DIAL for third control period and its variance compared to authority approved figures. 6. Actual manpower expenditure for MIAL and variance from authority approved manpower expenditure for the third control period. 7. Justification provided by MIAL for actual manpower expenses submitted by MIAL for third control period and the reason for variance with authority approved figures. 8. Examination and rationale provided by authority for actual manpower expenditure submitted by only MIAL for third control period and its variance compared to authority approved figures. 9. Using the rationale extracted in steps 4, 5, 7, and 8, suggest how the rationale or justification provided by BIAL in the MDA document for manpower expenditure for the third control period can be enhanced. For every suggestion made, give specific reason why the suggestion was made by you using relevant references from DIAL and MIAL tariff orders or consultation papers.",
            },
            "Utility Analysis": {
                "Analysis of electricity expenditure projection for BIAL for third control period": f"The uploaded document proposes the following: '{{document_summary}}'. Use the following steps for analysing the electricity expenditure projected by BIAL: 1- Year on Year actual growth of power consumption cost for third control period  by BIAL in Utlities cost_MDA document . 2-   Year on Year  actual power consumption by BIAL in the Utlities cost_MDA document. 3- Year on Year actual recoveries of power consumption by BIAL for third control period  in the Utlities cost_MDA document. 4- Justification provided by BIAL for the power expense  and the variance of power expense with authority approved figures in third control period in the Utlities cost_MDA document. 5- Year on Year growth of actual power expense submitted by DIAL for true up of third control period in the fourth control period consultation paper. 6- Year on Year actual growth of power consumption submitted by DIAL for third control period in the fourth control period consultation paper. 7- Year on Year actual recoveries from sub-concessionaries submitted by DIAL for third control period in the fourth control period consultation paper. 8- Justification for actual power expense in third control period provided by DIAL and the variance with authority approved figures in fourth control period consultation paper. 9- Examination and rationale provided by authority on actual power cost and consumption submitted by DIAL for third control period in the fourth control period consultation paper.  10- Year on Year growth of actual Electricity cost(utility expenses) submitted by MIAL for true up of third control period in the fourth control period consultation paper. 11- Year on Year  electricity  gross consumption(utlity expenses) submitted by MIAL for true up of third control period in the fourth control period consultation paper. 12- Year on Year  recoveries of electricity consumption submitted by MIAL for the trueup of third control period in the fourth control period consultation paper. 8 Justification for actual electricity cost for the true up of third control period provided by MIAL in the fourth control period consultation paper and the variance with authority approved figures. 9- Examination and rationale provided by authority on actual Electricity cost and consumption submitted by MIAL true of third control period in the fourth control period consultation paper.15- Using the rationale extracted in steps 4, 8, 9,13 and 14 suggests how the rationale or justification provided by BIAL in the MDA document for electricity cost  for third control period can be enhanced. For every suggestion made, give specific reason why the suggestion was made using relevant references from DIAL and MIAL tariff orders or consultation papers. when asked about MIAL only give information relevant to MIAL not DIAL Strictly.",
                "Analysis of water expenditure projection for BIAL for third control period": f"The uploaded document proposes the following: '{{document_summary}}'. Use the following steps for analysing the water expenditure projected by BIAL: 1- Year on Year actual  water expense by BIAL for third control period in the Utilities cost_MDA document MDA document. 2- Year on Year Water consumption by BIAL of true up for third control period in the Utilities cost_MDA document. 3- Year on Year actual recoveries of  water consumption by BIAL for the third control period in the Utilities cost_MDA document. 4- Justification provided by BIAL for the water cost for third control period and the variance of water expense with authority approved figures in third control period in the Utilities cost_MDA document. 5- Year on Year  water gross charge submitted by DIAL for true up of third control period in the fourth control period consultation paper. 6- Year on Year actual growth of water consumption submitted by DIAL for third control period in the fourth control period consultation paper. 7- Year on Year actual recoveries from sub- concessionaire submitted by DIAL for third control period in the fourth control period consultation paper. 8- Justification for actual  gross water charge  in third control period in the fourth control period consultation paper provided by DIAL and the variance with authority approved figures. 9- Examination and rationale provided by authority on actual water gross charge and consumption submitted by DIAL for third control period in the fourth control period consultation paper.  10- Year on Year actual water expense(utility expenses) submitted by MIAL for true up of third control period in the fourth control period consultation paper. 11- Year on Year actual water consumption(Kl) submitted by MIAL for true up of third control period in the fourth control period consultation paper. 12- Year on Year actual recoveries(kl) of water consumption submitted by MIAL for true up of the  third control period in the fourth control period consultation paper. 8- Justification for actual water gross amount for third control period in the MIAL fourth control period consultation paper provided by MIAL and the variance with authority approved figures. 9- Examination and rationale provided by authority on actual water gross amount  and consumption submitted by MIAL for third control period in the fourth control period consultation paper.15- Using the rationale extracted in steps 4, 8, 9,13 and 14 suggest how the rationale or justification provided by BIAL in the MDA document for water expenditure for trueup of  third control period can be enhanced. For every suggestion made, give specific reason why the suggestion made using relevant references from DIAL and MIAL tariff orders or consultation papers.when asked about MIAL only give information relevant to MIAL not DIAL Strictly",
            },
            # Add more use cases and their prompts here
            # "Another Analysis Type": {
            #     "Prompt 1 for another type": "Template for prompt 1 with '{document_summary}'",
            #     "Prompt 2 for another type": "Template for prompt 2 with '{document_summary}'"
            # }
        }

        # Select the analysis type
        st.session_state.mda_analysis_type = st.selectbox(
            "Select Analysis Type:",
            list(analysis_prompts_config.keys()),
            key="mda_analysis_type_selector",
        )

        uploaded_word_file_main = st.file_uploader(
            "Upload a Word Document (.docx)", type="docx", key="word_uploader_main"
        )

        if uploaded_word_file_main is not None:
            st.markdown("---")
            if st.button(
                "Generate Full Analysis Report",
                key="analyze_word_doc_button",
                type="primary",
            ):
                report_placeholder = st.empty()
                with st.spinner(
                    "Processing document and generating full report... This may take a moment."
                ):
                    extracted_text = extract_text_from_docx(uploaded_word_file_main)

                    if not extracted_text:
                        st.error(
                            "Could not extract text from the document. Please check the file and try again."
                        )
                        st.stop()

                    report_parts = []
                    report_parts.append(
                        f"## Analysis Report for: *{html.escape(uploaded_word_file_main.name)}*"
                    )
                    report_parts.append(
                        "This report analyzes the content of the uploaded MYTP File by comparing its key points against historical data ."
                    )

                    # Step 1: Generate Context Summary
                    st.write("Step 1/3: Generating context from uploaded document...")
                    summary_prompt = f"Please provide a concise, neutral summary of the key points in the following document text. This will be used as the context for historical analysis:\n\n---\n{extracted_text[:20000]}\n---"
                    try:
                        summary_response = (
                            synthesis_openai_client.chat.completions.create(
                                model=DEPLOYMENT_ID_VAL,
                                messages=[{"role": "user", "content": summary_prompt}],
                                temperature=0.1,
                                max_tokens=16000,
                            )
                        )
                        document_summary = summary_response.choices[0].message.content
                        report_parts.append(
                            "### 1. Document Context (Generated Summary)"
                        )
                        report_parts.append(document_summary)
                    except Exception as e:
                        st.error(f"Failed to generate document summary: {e}")
                        document_summary = None

                    if document_summary:
                        # Step 2: Draw Rationale from Historical Data
                        st.write("Step 2/3: Drawing rationale from historical data...")
                        report_parts.append("### 2. Rationale Based on Historical Data")

                        # Get prompts based on selected analysis type
                        analysis_prompts_for_type = analysis_prompts_config.get(
                            st.session_state.mda_analysis_type, {}
                        )

                        if not analysis_prompts_for_type:
                            st.warning(
                                f"No analysis prompts defined for '{st.session_state.mda_analysis_type}'. Please select a different type or add prompts to the configuration."
                            )
                        else:
                            for (
                                title,
                                prompt_template,
                            ) in analysis_prompts_for_type.items():
                                st.write(f"    - Analyzing: {title}")
                                # Format the prompt template with the document summary
                                full_prompt = prompt_template.format(
                                    document_summary=document_summary
                                )

                                analysis_answer = generate_answer_from_search(
                                    user_question=full_prompt,
                                    index_name=st.session_state.conv_agent_selected_index,
                                    use_hybrid_semantic=st.session_state.conv_agent_use_hybrid,
                                    vector_field=st.session_state.conv_agent_vector_field,
                                    semantic_config=st.session_state.conv_agent_semantic_config,
                                    temperature=st.session_state.conv_agent_temp,
                                    max_tokens_param=st.session_state.conv_agent_max_tokens,
                                    client_for_synthesis=synthesis_openai_client,
                                    show_details=False,  # Suppress verbose output for report generation
                                )
                                report_parts.append(f"#### {title}\n{analysis_answer}")

                        # Step 3: Final Report Generation
                        st.write("Step 3/3: Compiling final report...")
                        final_report_markdown = "\n\n---\n\n".join(report_parts)
                        report_placeholder.markdown(
                            final_report_markdown, unsafe_allow_html=True
                        )
                        st.success("Full analysis report generated successfully!")

    st.markdown("---")
    st.caption(" BIAL Multi-Agent Regulatory Platform | BIAL")


if __name__ == "__main__":
    try:
        main_app_logic()
    except Exception as e:
        st.error(f"An critical unexpected error occurred in the application: {e}")
        st.exception(e)
        print("--- Critical Unhandled Exception Caught by Global Handler ---")
        traceback.print_exc()
        print("--------------------------------------------------------")
