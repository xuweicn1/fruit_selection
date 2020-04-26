import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# 数据集地址
data_path = './datasets/fruit_selection.csv'


def parse_csv(line):
    """ 解析csv文件 """
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


train_dataset = tf.data.TextLineDataset(data_path)


# 跳过第一行
train_dataset = train_dataset.skip(1)

# 解析
train_dataset = train_dataset.map(parse_csv)

# 洗牌
train_dataset = train_dataset.shuffle(buffer_size=1000)

# 每次训练样本数
train_dataset = train_dataset.batch(32)


# 创建模型

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(3)
])


# 定义损失函数
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# 定义梯度
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)


# 优化随机梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 误差 和准确率
train_loss_results = []
train_accuracy_results = []

# 开始训练
num_epochs = 201
for epoch in range(num_epochs):
	epoch_loss_avg = tfe.metrics.Mean()
	epoch_accuracy = tfe.metrics.Accuracy()

	for x, y in train_dataset:
		grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
		epoch_loss_avg(loss(model, x, y))  
		epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

	train_loss_results.append(epoch_loss_avg.result())
	train_accuracy_results.append(epoch_accuracy.result())

	if epoch % 50 == 0:
		print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()
