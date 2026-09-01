import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import os


def find_common_substring(strings: list[str]):
    """Formating helper function to find the common
    substring in the list of strings to be used as
     a name of the trajectory"""
    if len(strings) > 1:
        strings = strings[1:]
    try:
        common_string = os.path.commonprefix(strings)
    except TypeError:
        common_string = None
    finally:
        if common_string[-1] == "_":
            common_string = common_string[:-1]
        if common_string[:7] == "ripper_":
            common_string = common_string[7:]
        return common_string


def filter_dataframe_by_name_column(
    df: pd.DataFrame, names: list[str]
) -> pd.DataFrame:
    keys_in_order: list = [None] + names
    df = df[df.name.apply(lambda x: x in keys_in_order)]
    df = df.set_index("name")
    df = df.reindex(keys_in_order)
    df.reset_index(drop=True)
    return df


def pxplot(
    data: pd.DataFrame,  # actuall data of peptides
    trajectories_names: list[list[str]],
    # list of trajectorites of names of the rows to plot, if None show all
    x_axis: str = "diversity",
    y_axis: str = "complexity",
    hover: str = "name",
    title: str = "",
    xaxis_title: str = "default",
    yaxis_title: str = "default",
    font_size: int = 14,
    fig_size: tuple[int, int] = (600, 600),
    show_peptides_in_plot: bool = False,
):
    if xaxis_title == "default":
        xaxis_title = x_axis
    if yaxis_title == "default":
        yaxis_title = y_axis

    fig = px.line()
    for trajectory in trajectories_names:
        df = filter_dataframe_by_name_column(data, trajectory)

        def _sequences() -> list[str]:
            """Formating the string to be printed in the plot"""
            new_list = [""]
            for experiment_key in trajectory:
                peptide_list = list(df.loc[experiment_key].observed_nodes)
                cnt = 0
                new_nice_string = ""
                for peptide in peptide_list:
                    if cnt % 6 == 5:
                        new_nice_string += "<br>"
                    new_nice_string += f"{peptide}, "
                    cnt += 1
                new_list.append(new_nice_string[:800])
            return new_list

        list_to_print = _sequences() if show_peptides_in_plot else None
        name_of_trajecotry = find_common_substring(trajectory)

        fig.add_trace(
            go.Scatter(
                x=df[x_axis],
                y=df[y_axis],
                name=name_of_trajecotry,
                mode="lines+markers",
                text=list_to_print,
            )
        )
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        font_family="Arial",
        font_color="black",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font_size=font_size,
        title=title,
    )
    return fig


def dictplot(
    data: dict[str, list[str]],  # actuall data of peptides
    keys: list[list[str]],  # how the trajectories are organised
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    font_size: int = 14,
    fig_size: tuple[int, int] = (600, 600),
    show_peptides_in_plot: bool = False,
):
    fig = px.line()
    for index, trajectory_keys in enumerate(keys):

        def _sequences() -> list:
            """Formating the string to be printed in the plot"""
            new_list = [""]
            for trajectory_key in trajectory_keys:
                cnt = 0
                new_nice_string = ""
                for peptide in data[trajectory_key]:
                    if cnt % 6 == 5:
                        new_nice_string += "<br>"
                    new_nice_string += f"{peptide}, "
                    cnt += 1
                new_list.append(new_nice_string[:1200])
            return new_list

        list_to_print = _sequences() if show_peptides_in_plot else None
        # list_to_print = None

        def _formated_name() -> str:
            name = f"{trajectory_keys[0]}"
            if name.startswith("ripper_"):
                name = name[7:]
            return name

        fig.add_trace(
            go.Scatter(
                x=[j for j in range(len(trajectory_keys) + 1)],
                y=[0] + [len(data[k]) for k in trajectory_keys],
                name=_formated_name(),
                mode="lines+markers",
                text=list_to_print,
            )
        )

    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        font_family="Arial",
        font_color="black",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font_size=font_size,
        title=title,
    )

    return fig
