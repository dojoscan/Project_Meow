import parameters as p
import tools as t

rp, i = t.get_last_ckpt(p.PATH_TO_CKPT_TEST)

print(i)
print(rp)