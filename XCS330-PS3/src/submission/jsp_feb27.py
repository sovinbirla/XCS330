import torch

n = 100
s = 100
a1 = torch.randint(s,size=(n,2))
a2 = torch.randint(s,size=(n,2))

D = [torch.sqrt((a1[i][0] - a2[i][0])^2 + (a1[i][1] - a2[i][1])^2) / 2 for i in range(n)] #radius of two original points
m = abs(a2-a1)/2

print()

area_outside = []
# for each midpoint, sample n points inside the circles
for i, (m_x, m_y) in enumerate(m):
    #for each midpoint, generate n thetas, figure out how many coordinates fall outside of s (0,0), (0,100), (100,0), (100,100)
    #polar coords
    temp_set = torch.rand(n)
    r = D * temp_set
    thetas = temp_set * 2 * torch.pi
    p_outside = 0
    for theta in thetas:
        x = m_x + r[i] * torch.cos(theta)
        y = m_y + r[i] * torch.sin(theta)
        if x <= 0 or x > 100 or y <= 0 or y > 100:
            p_outside+=1

    area_outside.append(p_outside/n)
        
 

print(R[:5], m[:5], area_outside[:5])
