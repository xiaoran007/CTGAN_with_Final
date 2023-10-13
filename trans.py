import pandas as pd

def creat():
    data = {'Normal_test_accuracy': [],
            'Normal_train_accuracy': [],
            'Normal_f1_score': [],
            }
    df = pd.DataFrame(data)
    return df


df = creat()
print(df)
add_df = {'Normal_test_accuracy': [1],
          'Normal_train_accuracy': [2],
          'Normal_f1_score': [1.5],
          }

add_df = pd.DataFrame(add_df)
print(add_df)
df = pd.concat([df, add_df], ignore_index=True)
print(df)
