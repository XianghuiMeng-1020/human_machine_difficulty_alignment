import transformers
names = ["Qwen2VLForConditionalGeneration", "InternVLForConditionalGeneration", "SmolVLMForConditionalGeneration", "AutoModelForImageTextToText"]
for n in names:
    print(n, n in dir(transformers))
