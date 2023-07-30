import matplotlib.pyplot as plot

def visualise(trajectory):
    fig = plot.figure()
    colours = ["r","b","g","y","m","c"]
    ax = fig.add_subplot(1,1,1, projection="3d")
    max_range = 0

    for index, selected_body in enumerate(trajectory):
        max_x = max(selected_body["x"])
        max_y = max(selected_body["y"])
        max_z = max(selected_body["z"])
        max_dimension = max(max_x, max_y, max_z)
        if max_dimension > max_range:
            max_range = max_dimension

        ax.plot(selected_body["x"], selected_body["y"], selected_body["z"], c = colours[index%len(colours)], label = selected_body["name"])

    ax.set_xlim([-max_range,max_range])
    ax.set_ylim([-max_range,max_range])
    ax.set_zlim([-max_range,max_range])

    plot.show()