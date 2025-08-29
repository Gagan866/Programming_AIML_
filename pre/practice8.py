# import matplotlib.pyplot as plt
# import numpy as np

# xpoints = np.array([1, 2, 6, 8])
# ypoints = np.array([3, 8, 1, 10])

# plt.plot(xpoints, ypoints,marker="_")
# plt.show()

import numpy as np 
import matplotlib.pyplot as plt 

# X-axis values 
x = [1, 2, 3, 4, 5] 

# Y-axis values 
y = [1, 4, 9, 16, 25] 

# Function to plot 
plt.plot(x, y) 

# Function add a legend 
plt.legend(['single element']) 

# function to show the plot 
plt.show()


# **Markers**
# 'o'	Circle	
# '*'	Star	
# '.'	Point	
# ','	Pixel	
# 'x'	X	
# 'X'	X (filled)	
# '+'	Plus	
# 'P'	Plus (filled)	
# 's'	Square	
# 'D'	Diamond	
# 'd'	Diamond (thin)	
# 'p'	Pentagon	
# 'H'	Hexagon	
# 'h'	Hexagon	
# 'v'	Triangle Down	
# '^'	Triangle Up	
# '<'	Triangle Left	
# '>'	Triangle Right	
# '1'	Tri Down	
# '2'	Tri Up	
# '3'	Tri Left	
# '4'	Tri Right	
# '|'	Vline	
# '_'	Hline	


# **Line Reference**

# Line Syntax	Description
# '-'	Solid line	
# ':'	Dotted line	
# '--'	Dashed line	
# '-.'	Dashed/dotted line


# **Color Reference**
# Color Syntax	Description
# 'r'	Red	
# 'g'	Green	
# 'b'	Blue	
# 'c'	Cyan	
# 'm'	Magenta	
# 'y'	Yellow	
# 'k'	Black	
# 'w'	White


# ...
# plt.plot(ypoints, marker = 'o', ms = 20, mec = 'hotpink', mfc = 'hotpink')
# ...


# **Line Style**
# Style	Or
# 'solid' (default)	'-'	
# 'dotted'	':'	
# 'dashed'	'--'	
# 'dashdot'	'-.'	
# 'None'	'' or ' '


# **Line Color**
# plt.plot(ypoints, color = 'r')


# **Labels**
# plt.title("Sports Watch Data", loc = 'left')
# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")


# **Grid**
# plt.grid(axis = 'y')
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)


# **Subplot**
# plt.subplot(1, 2, 2)



# **Scatter**
# plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')


# **Bar Graph**
# plt.barh(x, y, height = 0.1)
# plt.bar(x, y, width = 0.1)