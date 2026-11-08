from datasets import load_dataset

dataset = load_dataset("truthful_qa", "generation")

print(dataset)

train = dataset["validation"]

print(train.column_names)       
print(train[0])