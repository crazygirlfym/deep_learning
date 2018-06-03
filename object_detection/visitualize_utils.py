import matplotlib.pyplot as plt


def draw_attention(activation_map, input_length, output_length, predicted_text, text_,  output=None):
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)
    # activation_map = [[0.2,0.3,0.5], [0.7, 0.2, 0.1]]
    i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')
    # add colorbar

    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Probability', labelpad=2)
    # output_length = 3

    # predicted_text = ["this", "is", "test"]
    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    # input_length = 2
    # text_ = ["这是", "测试"]
    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()
    # ax.legend(loc='best')
    f.show()
    if(output != None):
        f.savefig(output+ ".pdf", bbox_inches='tight')