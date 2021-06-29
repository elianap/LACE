class GlobalExplanation:
    def __init__(
        self,
        global_expl_attribute,
        global_expl_attribute_values,
        global_expl_rules,
        ids_to_explain=[],
        classes_names=None,
        infos=None,
    ):
        self.ids_explained = ids_to_explain
        self.classes_names = classes_names
        self.global_expl_attribute = global_expl_attribute
        self.global_expl_attribute_values = global_expl_attribute_values
        self.global_expl_rules = global_expl_rules
        self.infos = infos

    def getGlobalExpl(self, expl_type, class_name=None):
        if expl_type == "attr":
            return self.getGlobalExplAttribute(class_name=class_name)
        elif expl_type == "attr_value":
            return self.getGlobalExplAttributeValues(class_name=class_name)
        elif expl_type == "rules":
            return self.getGlobalExplRules(class_name=class_name)
        else:
            raise ValueError(f"Admitted: attr, attr_value, rules, inserted {expl_type}")

    def getGlobalExplAttribute(self, class_name=None):
        if class_name:
            self._checkAvailable(self.global_expl_attribute, class_name)
            return self.global_expl_attribute[class_name]
        return self.global_expl_attribute

    def getGlobalExplAttributeValues(self, class_name=None):
        if class_name:
            self._checkAvailable(self.global_expl_attribute_values, class_name)
            return self.global_expl_attribute_values[class_name]
        return self.global_expl_attribute_values

    def getGlobalExplRules(self, class_name=None):
        if class_name:
            self._checkAvailable(self.global_expl_rules, class_name)
            return self.global_expl_rules[class_name]
        return self.global_expl_rules

    def _checkAvailable(self, dict_ofI, class_name):
        if class_name not in dict_ofI:
            if self.classes_names and class_name not in self.classes_names:
                raise ValueError(
                    f"Class {class_name} not in the dataset: {self.classes_names}"
                )
            raise ValueError(f"Class {class_name} not available")

    def plotGlobalExplanation(
        self, expl_type, interactive=True, target_class="total", sortedF=False
    ):
        global_expl_res = self.getGlobalExpl(expl_type, target_class)
        infos_t = " ".join(
            [
                f"{k}={self.infos[k]}"
                if k in self.infos and self.infos[k] is not None
                else ""
                for k in ["x", "d", "model"]
            ]
        )
        hovertext = None
        if expl_type == "rules":
            rule_mapping_id = {
                f"Rule_{i+1}": ", ".join(list(k)) for i, k in enumerate(global_expl_res)
            }
            x = list(global_expl_res.values())
            y = list(rule_mapping_id.keys())
            hovertext = list(rule_mapping_id.values())
        else:
            global_expl_res_s = global_expl_res
            if sortedF:
                global_expl_res_s = {
                    k: v
                    for k, v in sorted(
                        global_expl_res_s.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

            y = list(global_expl_res_s.keys())
            x = list(global_expl_res_s.values())

        xaxis_title = (
            f"Δ target class={target_class}" if target_class != "total" else ""
        )
        if interactive:
            from src.utils_plot import barPlotPlotly

            fig = barPlotPlotly(
                x, y, infos_t, xaxis_title=xaxis_title, hovertext=hovertext
            )
        else:
            from src.utils_plot import barPlotMPL

            fig = barPlotMPL(x, y, infos_t, xaxis_title=xaxis_title)
        if expl_type == "rules":
            print(rule_mapping_id)
        return fig

    def plotGlobalExplanation_v2(
        self,
        expl_type,
        interactive=True,
        target_class="total",
        sortedF=False,
        firstK=None,
    ):
        self_res = self.getGlobalExpl(expl_type, target_class)
        infos_t = " ".join(
            [
                f"{k}={self.infos[k]}"
                if k in self.infos and self.infos[k] is not None
                else ""
                for k in ["x", "d", "model"]
            ]
        )
        hovertext = None
        if expl_type == "rules":
            rule_mapping_id = {
                f"Rule_{i+1}": ", ".join(list(k)) for i, k in enumerate(self_res)
            }
            x = list(self_res.values())
            y = list(rule_mapping_id.keys())
            hovertext = list(rule_mapping_id.values())
        else:
            self_res_s = self_res
            if firstK:
                self_res_s = {
                    k: v
                    for k, v in sorted(
                        self_res_s.items(),
                        key=lambda item: abs(item[1]),
                        reverse=True,
                    )
                }
                self_res_s = dict(list(self_res_s.items())[:firstK])
            if sortedF:
                self_res_s = {
                    k: v
                    for k, v in sorted(
                        self_res_s.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

            y = list(self_res_s.keys())
            x = list(self_res_s.values())

        xaxis_title = (
            f"Δ target class={target_class}" if target_class != "total" else ""
        )
        if interactive:
            from src.utils_plot import barPlotPlotly

            fig = barPlotPlotly(
                x, y, infos_t, xaxis_title=xaxis_title, hovertext=hovertext
            )
        else:
            from src.utils_plot import barPlotMPL

            fig = barPlotMPL(x, y, infos_t, xaxis_title=xaxis_title)
        if expl_type == "rules":
            print(rule_mapping_id)
        return fig


# TODO
def interactiveGlobalExplanation(lace_explainer, global_explanation):
    import ipywidgets as widgets
    from IPython.display import display

    from copy import deepcopy
    from ipywidgets import HBox

    w_target = widgets.Select(
        options=global_explanation.global_expl_attribute.keys(),
        description="Target",
        disabled=False,
    )
    type_global = {
        "attribute": "attr",
        "attribute values": "attr_value",
        "rules": "rules",
    }
    w_type = widgets.Select(
        options=type_global.keys(), description="Type", disabled=False
    )
    h1 = HBox([w_type, w_target])
    display(h1)
    sa = {}

    def clearAndShow(btNewObj):
        from IPython.display import clear_output

        clear_output()
        display(h1)
        display(h)

    def getValuesPlotExplanation(btn_object):

        fig = global_explanation.plotGlobalExplanation(
            type_global[w_type.value], interactive=True, target_class=w_target.value
        )
        fig.show()

    btnGlobalExpl = widgets.Button(description="Plot global explanation")
    btnGlobalExpl.on_click(getValuesPlotExplanation)
    btnNewSel = widgets.Button(description="New selection")
    btnNewSel.on_click(clearAndShow)
    h = HBox([btnGlobalExpl, btnNewSel])
    display(h)


"""
def barPlotPlotly(x, y, title, reverse=True, xaxis_title="", hovertext=None):
    import plotly.graph_objects as go

    if reverse:
        x.reverse()
        y.reverse()
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker=dict(color="#bee2e8", line=dict(color="black", width=0.4)),
            hovertext=hovertext,
            hoverlabel=dict(namelength=-1, font_size=10),
        )
    )

    axis = dict(
        showline=False,
        zeroline=False,
        showgrid=False,
    )

    fig.update_layout(
        title={"text": title},
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode="closest",
        plot_bgcolor="rgb(248,248,248)",
        width=500,
        height=350,
        xaxis=axis,
        yaxis=axis,
    )
    fig.update_xaxes(title_text=xaxis_title)
    return fig


def barPlotMPL(x, y, title, reverse=True, xaxis_title=""):
    import matplotlib.pyplot as plt

    fig, pred_ax = plt.subplots(1, 1)

    pred_ax.set_title(title)
    pred_ax.set_xlabel(xaxis_title)

    pred_ax.barh(
        y,
        width=x,
        align="center",
        color="#bee2e8",
        linewidth="1",
        edgecolor="black",
    )
    if reverse:
        pred_ax.invert_yaxis()
    return fig

"""