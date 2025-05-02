import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def read_data(file_path):
    sentences = []
    current_sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split(' ', 1)  # Split into word and tag
                if len(parts) == 2:
                    word, tag = parts
                    current_sentence.append((word, tag))
        if current_sentence:
            sentences.append(current_sentence)
    return sentences


def build_vocab(sentences):
    word_counts = {}
    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word in word_counts:
        word_to_ix[word] = len(word_to_ix)
    return word_to_ix


class NERDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_indices = [self.word_to_ix.get(word, self.word_to_ix['<UNK>']) for word, _ in sentence]
        tag_indices = [self.tag_to_ix[tag] for word, tag in sentence]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)  # Ignore index
    return inputs_padded, targets_padded, lengths


# Read all sentences
file_path = "ner_dataset.txt"
sentences = read_data(file_path)

# Split into train (70%), temp (30%)
train_sentences, temp_sentences = train_test_split(sentences, test_size=0.3,
                                                   random_state=42)

# Split temp into validation (50%) and test (50%) (each 15% of total)
val_sentences, test_sentences = train_test_split(
    temp_sentences, test_size=0.5, random_state=42
)

# Create vocabs
tag_to_ix = {'O': 0, 'I-LOC': 1, 'I-PER': 2, 'I-ORG': 3, 'I-MISC': 4}
word_to_ix = build_vocab(train_sentences)

# Create datasets
train_dataset = NERDataset(train_sentences, word_to_ix, tag_to_ix)
val_dataset = NERDataset(val_sentences, word_to_ix, tag_to_ix)
test_dataset = NERDataset(test_sentences, word_to_ix, tag_to_ix)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)
