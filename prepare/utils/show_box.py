from matplotlib import pyplot as plt


def show_boxes(im_path, dets, cls, colors=None):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(dets))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):

        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i], linewidth=1.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{}'.format(cls[i]),
                bbox=dict(facecolor=colors[i], alpha=0.5),
                fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
    plt.show()
