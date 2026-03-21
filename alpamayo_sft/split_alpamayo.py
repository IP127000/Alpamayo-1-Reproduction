from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model_path = "/mnt/alpamayo1_5"
save_path = "/mnt/alpamayo1_5_vlm"
model = Alpamayo1_5.from_pretrained("model_path", dtype="bfloat16")
processor=helper.get_processor(model.tokenizer)
model.vlm.save_pretrained(save_path)
processor.save_pretrained(save_path)

