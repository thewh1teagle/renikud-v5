from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("אז את רוצה את זה? או שאתה רוצה את זה? מה אתם אומרים המודל עובד טוב?"))
