from jesnk_utils.telebot import Telebot


import matplotlib.pyplot as plt

telebot = Telebot()

# make a figure and axes with dimensions as desired.
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.set_title('Title')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.text(0.2, 0.5, 'Text', fontsize=18, color='r')

# call the function to show the plot.
# plt.show()

# save png
plt.savefig('test.png')
telebot.send_image('./test.png')
