import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

"""example of reading pkl files"""

clamp_type = "ends"
training_case = 1
BDLO_type = 1

train_loss_1 = np.array(pd.read_pickle(r"../training_record/train_%s_loss_TreeLSTM_%s_%s.pkl" % (clamp_type, training_case, BDLO_type)))
train_step_1 = np.array(pd.read_pickle(r"../training_record/train_%s_step_TreeLSTM_%s_%s.pkl" % (clamp_type, training_case, BDLO_type)))
print("Train loss minimum: ", np.min(np.sqrt(train_loss_1)))

eval_loss_1 = np.array(pd.read_pickle(r"../training_record/eval_%s_loss_TreeLSTM_%s_%s.pkl" % (clamp_type, training_case, BDLO_type)))
eval_step_1 = np.array(pd.read_pickle(r"../training_record/eval_%s_epoches_TreeLSTM_%s_%s.pkl" % (clamp_type, training_case, BDLO_type)))

eval_rmse = np.sqrt(eval_loss_1)
best_idx = np.argmin(eval_rmse)

best_eval_rmse = eval_rmse[best_idx]
best_eval_step = eval_step_1[best_idx]

print(f"Best eval RMSE: {best_eval_rmse:.6f}")
print(f"Best eval at iteration: {best_eval_step} (index={best_idx})")

# print(np.sqrt(eval_loss_1))
print("Eval loss minimum: ", np.min(np.sqrt(eval_loss_1)))
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(10)
fig.set_figwidth(20)

line1 = ax1.plot(train_step_1, np.sqrt(train_loss_1), label='%s'%BDLO_type)
line2 = ax2.plot(eval_step_1, np.sqrt(eval_loss_1), label='%s'%BDLO_type)

# # # #
ax1.set_title('BDLO1: Training')
ax1.set_xlabel('Training Iterations')
ax1.set_ylabel('Square Root of MSE')

ax2.set_title('BDLO1: Eval')
ax2.set_xlabel('Training Iterations')
ax2.set_ylabel('Square Root of MSE')

ax1.grid(which = "minor")
ax1.minorticks_on()
ax2.grid(which = "minor")
ax2.minorticks_on()
plt.legend()

save_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir,
    f"TreeLSTM_loss_bdlo{BDLO_type}_{clamp_type}_case{training_case}.png"
)

fig.tight_layout()
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()


