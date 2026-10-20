"""W&B saved view with the Comet sections, metrics and Russian chart titles.

Run directly after `wandb sync` of an offline recording to create the view
without starting training: python numbergrid_wandb_dashboard.py --project numbergrid
"""
import argparse
import json
from math import ceil
import os

from numbergrid_comet_dashboard import (CAMPAIGN_CHARTS, CHART_TITLES, ROLLOUT_CHARTS,
                                        TECHNICAL_SECTIONS, VIEW_NAME)

# Stable ?nw= key shaped like generated view names: every run updates one
# project view in place instead of adding duplicates, like the Comet view.
VIEW_KEY = 'disciplesii'
X_AXIS = 'training_steps'
OWN_SECTIONS = ('rollout', 'campaign')


def view_sections(existing=False):
    """(name, metrics, columns, is_open) in the Comet order.

    An existing view keeps its other sections, as Comet does; the collapsed
    technical sections are added only to a new view.
    """
    own = [('rollout', ROLLOUT_CHARTS, 2, True), ('campaign', CAMPAIGN_CHARTS, 3, True)]
    if existing:
        return own
    return own + [(name, metrics, 3, False) for name, metrics in TECHNICAL_SECTIONS]


def wandb_section(name, metrics, columns, is_open):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    return ws.Section(
        name=name, is_open=is_open,
        panels=[wr.LinePlot(title=CHART_TITLES.get(metric, metric), x=X_AXIS, y=[metric])
                for metric in metrics],
        # The standard W&B layout paginates; enough rows keep every chart of
        # the section on one page, as in Comet.
        layout_settings=ws.SectionLayoutSettings(columns=columns, rows=ceil(len(metrics) / columns)),
        panel_settings=ws.SectionPanelSettings(x_axis=X_AXIS))


def configure_wandb_dashboard(entity, project):
    """Create or update the project's view; return its tracking metadata."""
    import wandb_workspaces.workspaces as ws

    try:
        # from_url reads only the path and ?nw=; the API host comes from the
        # saved W&B login, so self-hosted servers work as well.
        workspace = ws.Workspace.from_url(f'https://wandb.ai/{entity}/{project}?nw={VIEW_KEY}')
    except ValueError as exc:  # pydantic errors of an unreadable view are ValueErrors too
        if 'not found' not in str(exc):
            raise
        workspace = None
    created = workspace is None
    sections = [wandb_section(*section) for section in view_sections(existing=not created)]
    if created:
        workspace = ws.Workspace(entity=entity, project=project, name=VIEW_NAME, sections=sections)
        workspace._internal_name = f'nw-{VIEW_KEY}-v'
    else:
        workspace.name = VIEW_NAME
        workspace.sections = sections + [section for section in workspace.sections
                                         if section.name not in OWN_SECTIONS]
    # A loaded view keeps its old panel generation and sorting flags; keep it
    # curated and in the Comet order like a newly created one.
    workspace._auto_generate_panels = False
    workspace.settings.sort_panels_alphabetically = False
    workspace.settings.x_axis = X_AXIS
    workspace.settings.smoothing_type = 'none'
    workspace.settings.smoothing_weight = 0
    workspace.save()
    return {'status': 'created' if created else 'updated', 'name': VIEW_NAME, 'view': VIEW_KEY,
            'url': workspace.url, 'x_axis': X_AXIS, 'first_section': 'rollout',
            'first_metric': ROLLOUT_CHARTS[0], 'campaign_charts': len(CAMPAIGN_CHARTS),
            'campaign_window_charts': sum(name.startswith('campaign/window/') for name in CAMPAIGN_CHARTS)}


def main():
    parser = argparse.ArgumentParser(description='Create or update the W&B "Disciples II" view')
    parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT', 'numbergrid'))
    parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY'),
                        help='Default: the entity of the saved W&B login')
    args = parser.parse_args()
    import wandb

    entity = args.entity or wandb.Api().default_entity
    print(json.dumps(configure_wandb_dashboard(entity, args.project), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
