import numpy as np

data_path = "../ptbxl_dl_dataset_v2/datasets/train_signals.npz"

with np.load(data_path, allow_pickle=True) as data:
    print("المفاتيح الأساسية داخل الملف:")
    keys = list(data.keys())
    print(keys)
    print()

    for key in keys:
        print(f"--- المفتاح: {key} ---")
        content = data[key]
        print("نوع المحتوى:", type(content))

        # الطباعة إذا كان عبارة عن structured array
        if hasattr(content, "dtype") and content.dtype.names:
            print("المحتوى عبارة عن structured array وهذه أسماء الحقول:")
            print(content.dtype.names)

        # إذا كان مصفوفة من dicts
        elif isinstance(content[0], dict):
            print("المحتوى عبارة عن مصفوفة من قواميس، هذه المفاتيح داخل أول قاموس:")
            for sub_key in content[0].keys():
                print(" -", sub_key)

        else:
            print("المحتوى ليس قاموسًا ولا structured array، مجرد ndarray.")

        print("الشكل (shape):", content.shape)
        print("نوع البيانات (dtype):", content.dtype)
        print()
