from difflib import Differ

A = "reboil|123|122|456|456"
B = "check|123|122|456|456"

a1 = ['reboil',123,122,456,456]
b1 = ['check',123,122,456,456]

tuple(Differ().compare(a1,b1))