import pandas as pd
a = pd.read_pickle("datos/panel/panel_2024.pkl")[["id_local","rotulo","norm"]]
b = pd.read_pickle("datos/panel/panel_2025.pkl")[["id_local","rotulo","norm"]]
m = a.merge(b, on="id_local", suffixes=("_24","_25"))
print(m[m.norm_24 != m.norm_25][["rotulo_24","rotulo_25"]].sample(30, random_state=1).to_string())