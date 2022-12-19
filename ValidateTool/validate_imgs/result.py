import pandas as pd

# read result
Hanover = pd.read_csv("Hanover_Hamburg/Hanover_Hamburg_color_vec.csv")
Nuremberg = pd.read_csv("Nuremberg_Hamburg/Nuremberg_Hamburg_color_vec.csv")

# count human_check == 1 and caculate accuracy
Hanover_HumanCheck = len(Hanover[Hanover['human_check'] == 1])
Nuremberg_HumanCheck = len(Nuremberg[Nuremberg['human_check'] == 1])

total = Hanover_HumanCheck + Nuremberg_HumanCheck
acc = total / (len(Hanover) + len(Nuremberg))

print(total)
print(len(Hanover) + len(Nuremberg))
print(acc)


