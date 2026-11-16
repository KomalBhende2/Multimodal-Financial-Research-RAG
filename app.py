from agents.orchestrator import ResearchOrchestrator


def main():

    print("=" * 60)
    print("Institutional Multimodal Research RAG")
    print("=" * 60)

    query = input("\nEnter your research query: ")

    orchestrator = ResearchOrchestrator()

    orchestrator.run(query)


if __name__ == "__main__":
    main()