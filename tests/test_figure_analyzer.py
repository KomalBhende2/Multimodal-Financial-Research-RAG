from pathlib import Path

from ingestion.analyzers.figure_analyzer import FigureAnalyzer

image = Path(
    "data/extracted_figures/page_17/page_1_figure_1.png"
)

analyzer = FigureAnalyzer()

result = analyzer.analyze(image)

print()
print("=" * 70)
print("FIGURE ANALYSIS")
print("=" * 70)

if result is None:

    print("Image not found!")
    print(image)

else:

    for key, value in result.items():

        print(f"{key}:")
        print(value)
        print()