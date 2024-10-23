'''
******************************************************************************************************************************************************

Ниже представлен полный рабочий код для распределенного обучения модели с использованием нескольких устройств (например, CPU или GPU) через PyTorch.
Этот код поддерживает распределенное обучение, динамическую настройку гиперпараметров, а также обучение с использованием токенизатора
и модели RobertaForMaskedLM. Он также включает примеры аргументов для запуска и базовые настройки для обучения на нескольких устройствах.

******************************************************************************************************************************************************
'''
import os
import json
import torch
import torch.distributed as dist
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd

# Чтение конфигурации из файла
with open('config.json') as config_file:
    config = json.load(config_file)

# Используем параметры из файла
inputs_path = config['inputs_path']
outputs_path = config['outputs_path']
model_path = config['model_path']
epochs = config['epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
world_size = config['world_size']

# Установим параметры адреса и порта мастера для распределенного обучения
os.environ['MASTER_ADDR'] = '172.22.128.1'  # IP адрес главной машины
os.environ['MASTER_PORT'] = '443'  # Порт для коммуникации

# Датасет для обучения
class TextDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = inputs  # Тексты без запятых (входные данные)
        self.outputs = outputs  # Тексты с запятыми (выходные данные)
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs.iloc[idx]
        output_text = self.outputs.iloc[idx]

        inputs = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        outputs = self.tokenizer(output_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        labels = outputs['input_ids'].squeeze(0)

        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }

# Функция для запуска обучения
def train(rank, world_size, args):
    print(f"Запуск на rank {rank}.")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Инициализация модели
    model = RobertaForMaskedLM.from_pretrained(args['model_path'])
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Токенизатор
    tokenizer = RobertaTokenizer.from_pretrained(args['model_path'])
    
    # Загружаем данные
    inputs = pd.read_csv(args['inputs_path'])['input']
    outputs = pd.read_csv(args['outputs_path'])['output']

    # Создаем датасет и загрузчик данных
    train_dataset = TextDataset(inputs, outputs, tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], sampler=train_sampler)

    # Оптимизатор
    optimizer = optim.Adam(ddp_model.parameters(), lr=args['learning_rate'])

    # Тренировка
    for epoch in range(args['epochs']):
        ddp_model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: val.to(rank) for key, val in batch.items()}
            outputs = ddp_model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")

    # Завершаем процесс группы
    dist.destroy_process_group()

# Основной блок кода
if __name__ == "__main__":
    # Загружаем конфигурацию из файла
    args = config

    world_size = args['world_size']
    processes = []

    # Запускаем процессы для распределенного обучения
    for rank in range(world_size):
        p = mp.Process(target=train, args=(rank, world_size, args))
        p.start()
        processes.append(p)

    # Ожидаем завершения всех процессов
    for p in processes:
        p.join()
