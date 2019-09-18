# few shot settings.
ways = 5
shots = 5
dev_shots = 5
queries = 5
dev_queries = 5
episodes = 100
dev_episodes = 30
num_labels = 33
role_dim = 100


# model hyperparameters.
hidden_dim = 100
update_embedding = True
dropout_pl = 0.8 # keep prob.
optimizer = "Adam"
lr_pl = 0.001 # 0.01
clip_grad = 5.0
shuffle = True
att_hidden_dim = 200
hidden_dim_match = 200
att_hidden_dim_match = 200
epoch = 100
pre_train_word_embedding = True
embedding_dim = 100
entity_embedding_dim = 100
pos_embedding_dim = 50

enc = 100

# save model.
model_path = ""
summary_path = ""
log_path = ""
result_path = ""



