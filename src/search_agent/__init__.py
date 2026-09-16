from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


load_dotenv()


class SearchSelection(BaseModel):
    title: str = Field(
        description="The title of the selected search result."
    )

    summary: str = Field(
        description="A short, factual summary of the selected search result."
    )

    source_index: int = Field(
        description=(
            "The number of the Tavily result supporting this item. "
            "It must match one of the numbered search results."
        )
    )


class SearchSelectionResponse(BaseModel):
    results: List[SearchSelection]


class SearchResult(BaseModel):
    title: str
    summary: str
    source: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
)

search = TavilySearch()

structured_llm = llm.with_structured_output(
    SearchSelectionResponse
)


def research(query: str) -> SearchResponse:
    """
    Search Tavily for the supplied query, select the most relevant
    results using the LLM, and return their titles, summaries,
    and source URLs.
    """

    search_results = search.invoke(
        {
            "query": query,
            "search_depth": "basic",
        }
    )

    results = search_results.get("results", [])

    if not results:
        return SearchResponse(results=[])

    numbered_results = "\n\n".join(
        (
            f"[{index}]\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )
        for index, result in enumerate(results, start=1)
    )

    selection = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research assistant. "
                    "Select the 5 most relevant search results for the user's query. "
                    "For each selected result, provide a clear title, "
                    "a factual short summary, and the source_index of the "
                    "Tavily result supporting it. "
                    "Use only the supplied search results. "
                    "Do not invent facts or source indexes. "
                    "Each source_index must match one of the numbered results."
                )
            ),
            HumanMessage(
                content=(
                    f"User query:\n{query}\n\n"
                    f"Numbered Tavily search results:\n{numbered_results}"
                )
            ),
        ]
    )

    final_results: List[SearchResult] = []

    for item in selection.results[:5]:
        result_index = item.source_index - 1

        if 0 <= result_index < len(results):
            source_url = results[result_index].get("url")

            if source_url:
                final_results.append(
                    SearchResult(
                        title=item.title,
                        summary=item.summary,
                        source=source_url,
                    )
                )

    return SearchResponse(results=final_results)


def main() -> None:
    query = input("What would you like to search for? ").strip()

    if not query:
        print("Please enter a search query.")
        return

    response = research(query)

    print("\n===== SEARCH RESULTS =====")

    if not response.results:
        print("No relevant results found.")
        return

    for index, result in enumerate(response.results, start=1):
        print(f"\n{index}. {result.title}")
        print(f"   {result.summary}")
        print(f"   Source: {result.source}")


if __name__ == "__main__":
    main()