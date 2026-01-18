import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel


def convert_sentence_transformer():
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.config.return_dict = False

    example_text = "Hello, world! This is a test sentence."
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            strict=False
        )

    traced_model.save("minilm_l12_v2_traced.pt")
    tokenizer.save_pretrained("minilm_tokenizer")
    print("Model saved as minilm_l12_v2_traced.pt")
    print("Tokenizer saved in minilm_tokenizer/")


def convert_dinov2():
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.config.return_dict = False

    dummy_image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_image, strict=False)

    traced_model.save("dinov2_base_traced.pt")
    processor.save_pretrained("dinov2_processor")

    print("DINOv2 model saved as dinov2_base_traced.pt")
    print("Image processor saved in dinov2_processor/")


if __name__ == '__main__':
    convert_sentence_transformer()
    convert_dinov2()