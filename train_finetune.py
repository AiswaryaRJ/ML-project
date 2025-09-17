from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# 1. Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load your dataset
df = pd.read_csv('career_pairs.csv')

# Convert to InputExamples
train_examples = [InputExample(texts=[row['description'], row['career']], label=1.0) for _, row in df.iterrows()]

# 3. Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 4. Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# 5. Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# 6. Save the fine-tuned model
model.save('fine_tuned_career_model')
print("✅ Fine-tuning complete! Model saved at 'fine_tuned_career_model'")
