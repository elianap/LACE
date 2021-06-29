def interactiveAttributeLevelInspection(
    lace_explainer,
    instance_dec,
    targetClass,
    encoders,
    class_col_name="class",
    dataset_name="",
):
    import ipywidgets as widgets
    from IPython.display import display

    from copy import deepcopy
    from ipywidgets import HBox

    feature_names = []
    attributes = lace_explainer.train_dataset.attributes()
    continuos_features = lace_explainer.train_dataset.continuos_attributes
    for f in [v[0] for v in attributes]:
        feature_names.append(f)
    w = widgets.SelectMultiple(
        options=feature_names, description="Feature", disabled=False
    )

    display(w)
    sa = {}

    def clearAndShow(btNewObj):
        from IPython.display import clear_output

        clear_output()
        display(w)
        display(btn)

    def getSlidersValues(btn_obj_expl):
        from copy import deepcopy

        inst1 = deepcopy(instance_dec)
        # inst1=dict(inst1.drop(class_col_name))
        for s in sa.values():
            inst1[s.description] = s.value
            inst1[s.description] = s.value

        from src.utils_analysis import encodeInstance, DiscretizeInstance

        instance_en = encodeInstance(inst1, encoders)
        instance_discr = DiscretizeInstance(
            inst1[feature_names], continuos_features, dataset_name=dataset_name
        )

        lace_explanation = lace_explainer.explain_instance(
            instance_en,
            targetClass,
            featureMasking=True,
            discretizedInstance=instance_discr,
            verbose=False,
        )
        lace_explanation.plotExplanation(showRuleKey=True, retFig=False)
        lace_explanation.local_rules.printLocalRules()

    # def compareOriginal(btnCompare):
    #         inst1=deepcopy(instance_dec)
    #         attr=[i.name for i in inst1.domain.attributes]
    #         n=""
    #         attr_name=[]
    #         for s in sa.values():
    #             if inst1[s.description]!=s.value:
    #               attr_name.append(s.description)
    #             inst1[s.description]=s.value
    #             n=n+"_"+s.description+"_"+s.value
    #         self.XPLAIN_explainer_o.comparePerturbed( str(self.n_inst)  ,str(inst1.id)+n, inst1, [attr.index(i) for i in attr_name])

    def getSlidersAttributes(btn_object):
        sa.clear()
        attr_values_dict = {v[0]: v[1] for v in attributes}
        for a in w.value:
            if attr_values_dict[a]:
                wa = widgets.SelectionSlider(
                    options=attr_values_dict[a],
                    description=a,
                    disabled=False,
                    value=instance_dec[a],
                    continuous_update=False,
                )
                sa[a] = wa
        for s in sa.values():
            display(s)
        btnExpl = widgets.Button(description="Get explanation")
        btnExpl.on_click(getSlidersValues)
        btnNewSel = widgets.Button(description="New selection")
        btnNewSel.on_click(clearAndShow)
        # btnCompare = widgets.Button(description='Compare original')
        # btnCompare.on_click(compareOriginal)
        h = HBox([btnExpl, btnNewSel])
        display(h)

    btn = widgets.Button(description="Select attributes")
    btn.on_click(getSlidersAttributes)
    display(btn)