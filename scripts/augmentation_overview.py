"""Overview of data augmentation techniques used across assignments."""


def list_augmentation_techniques():
    """Document augmentation techniques by assignment."""
    techniques = {
        "Assignment 3 (YOLO)": [
            "Random horizontal flip",
            "Random rotation (10 degrees)",
            "Mosaic augmentation",
            "Color jitter"
        ],
        "Assignment 5 (LSTM)": [
            "Sequence padding",
            "Random sequence truncation"
        ]
    }
    
    for assignment, augs in techniques.items():
        print(f"\n{assignment}:")
        for aug in augs:
            print(f"  - {aug}")


if __name__ == "__main__":
    list_augmentation_techniques()
