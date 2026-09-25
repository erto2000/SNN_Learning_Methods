"""Small shared safeguards for resumable validation searches."""
import ast
import hashlib
import json
from pathlib import Path
import optuna
from copy import deepcopy
from utils.console import line, section


_SIGNATURE_VERSION = 3
_OUTPUT_CALLS = {'print', 'panel', 'line', 'section', 'close', 'timestamp'}


class _WithoutConsoleOutput(ast.NodeTransformer):
    """Remove presentation-only code before fingerprinting experiment behavior."""

    def visit_ImportFrom(self, node):
        if node.module == 'utils.console':
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name) and call.func.id in _OUTPUT_CALLS:
                return None
            if (isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'set_verbosity'):
                return None
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not node.body and not node.orelse:
            return None
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'save_results':
            node.keywords = [kw for kw in node.keywords if kw.arg != 'quiet']
        return node


def _behavior_source(path: Path) -> bytes:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    tree = _WithoutConsoleOutput().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False).encode()


def _protocol_signature(config) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=lambda obj: {
        'type': type(obj).__name__, 'state': vars(obj) if hasattr(obj, '__dict__') else str(obj),
    }).encode())
    paths = [root / 'experiment_config.py', root / 'optuna_window_grid_two_phase.py',
             root / 'optuna_window_independent_two_phase.py']
    for folder in ('learners', 'networks', 'timeseries', 'utils'):
        paths.extend(sorted((root / folder).rglob('*.py')))
    # Neither console formatting nor the resume/manifest helper changes how a
    # search trial trains or evaluates a model. Excluding both also prevents
    # this fingerprint implementation from invalidating its own signature.
    paths = [path for path in paths if path.name not in {'console.py', 'optuna_support.py'}]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(_behavior_source(path))
    return digest.hexdigest()


def create_study(*, config, **kwargs):
    # A changed training protocol must not silently reuse previous observations.
    signature = _protocol_signature(config)
    study = optuna.create_study(**kwargs)
    previous = study.user_attrs.get('experiment_signature')
    version = study.user_attrs.get('experiment_signature_version')
    if previous is not None and (version != _SIGNATURE_VERSION or previous != signature):
        raise ValueError('Study configuration/code changed. Use a fresh Optuna database and results directory.')
    if study.trials and previous is None:
        raise ValueError('Study has trials without a protocol signature. Use a fresh Optuna database and results directory.')
    study.set_user_attr('experiment_signature', signature)
    study.set_user_attr('experiment_signature_version', _SIGNATURE_VERSION)
    study.set_user_attr('evaluation_split', 'validation')
    return study


def preferred_study(phase1, phase2):
    """Select by validation, preferring the refined phase when it succeeded."""
    for phase, study in ((2, phase2), (1, phase1)):
        if study is not None and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            return phase, study
    return None, None


def _comparable_final_config(config):
    """Return the behavior-relevant config used to identify a saved final run."""
    comparable = deepcopy(config)
    comparable.pop('TRANSFORM', None)
    comparable.pop('RUN_ID', None)
    selection = comparable.get('FINAL_TEST')
    if isinstance(selection, dict):
        selection.pop('experiment_signature', None)
    # JSON stores tuples as lists (for example TIME_EVAL_FRACS). Compare both
    # live and saved configurations in the same serialized representation.
    return json.loads(json.dumps(comparable, sort_keys=True))


def run_final_test(base_config, pipeline, *, study, phase, family, experiment,
                   dataset, learner, length, hop_ratio, results_dir):
    """Retrain a validation-selected configuration at the normal epoch budget."""
    from utils.runner import run_one
    from visualization.training_results import save_results
    if study is None:
        return None
    cfg = deepcopy(base_config)
    hop = max(1, round(length * hop_ratio))
    selection = dict(family=family, experiment=experiment, study=study.study_name,
                     trial=study.best_trial.number, phase=phase,
                     validation_score=study.best_value,
                     metric=cfg.get('OPTUNA_METRIC', 'sample_acc'),
                     experiment_signature=study.user_attrs.get('experiment_signature'))
    cfg.update(LEARNER=learner, TEST_EVERY_EPOCH=False,
               WINDOW=dict(L=int(length), hop=int(hop), hop_ratio=float(hop_ratio)),
               FINAL_TEST=selection)
    comparable_config = _comparable_final_config(cfg)
    identity_config = {k: v for k, v in cfg.items() if k != 'TRANSFORM'}
    identity = hashlib.sha256(json.dumps(identity_config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    cfg['RUN_ID'] = f'{dataset}-{learner}-final-{family}-{experiment}-{identity}'
    cfg['TRANSFORM'] = deepcopy(pipeline)
    root = Path(results_dir)
    summary_path = root / 'runs' / cfg['RUN_ID'] / 'summary.json'
    manifest_path = root / 'optuna' / 'final_tests' / family / f'{dataset}-{learner}-{experiment}.json'
    result = None
    # Prefer the manifest-selected run when its effective configuration is
    # unchanged. This survives fingerprint-format and console-only changes.
    if manifest_path.exists():
        try:
            manifest_saved = json.loads(manifest_path.read_text(encoding='utf-8'))
            prior_path = root / 'runs' / manifest_saved['run_id'] / 'summary.json'
            prior = json.loads(prior_path.read_text(encoding='utf-8'))
            if (prior.get('status') == 'ok'
                    and prior.get('final', {}).get('evaluation_split') == 'test'
                    and _comparable_final_config(prior.get('config', {})) == comparable_config):
                result = prior
                cfg['RUN_ID'] = manifest_saved['run_id']
                line(f"{experiment.title()} | Reusing saved test run")
        except (KeyError, TypeError, ValueError, OSError):
            pass
    if result is None and summary_path.exists():
        try:
            saved = json.loads(summary_path.read_text(encoding='utf-8'))
            if saved.get('status') == 'ok' and saved.get('final', {}).get('evaluation_split') == 'test':
                result = saved
                line(f"{experiment.title()} | Reusing saved test run")
        except (ValueError, OSError):
            pass
    if result is None:
        section(f"FINAL TEST | {experiment.upper()}")
        line(f"Train {cfg['EPOCHS']} epochs  |  L {length}  |  Hop {hop}")
        result = run_one(cfg)
        save_results([result], base_dir=results_dir, make_plots=True)
    manifest_dir = root / 'optuna' / 'final_tests' / family
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(run_id=cfg['RUN_ID'], status=result.get('status'),
                    dataset=dataset, learner=learner, **selection)
    (manifest_dir / f'{dataset}-{learner}-{experiment}.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    if result.get('status') != 'ok':
        line(f"FAILED | {result.get('error')} | Resume to retry")
    else:
        line(f"Test accuracy {result['final']['sample_acc']:.2f}%")
    return result


def export_final_tests(results_dir, out_dir, family):
    """Export only the currently selected test runs, never rank by test score."""
    import pandas as pd
    from timeseries.splitting import split_report_fields
    root = Path(results_dir)
    rows = []
    for manifest_path in sorted((root / 'optuna' / 'final_tests' / family).glob('*.json')):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        summary = json.loads((root / 'runs' / manifest['run_id'] / 'summary.json').read_text(encoding='utf-8'))
        cfg, final = summary['config'], summary.get('final', {})
        if not cfg.get('DATA_SPLIT'):
            continue
        theory = summary.get('memory', {}).get('theory', {})
        rows.append(dict(
            **split_report_fields(summary.get('meta', {})),
            dataset=manifest['dataset'], learner=manifest['learner'], experiment=manifest['experiment'],
            status=summary['status'], evaluation_split='test', epochs=cfg['EPOCHS'],
            win_L=cfg['WINDOW']['L'], hop=cfg['WINDOW']['hop'], hop_ratio=cfg['WINDOW']['hop_ratio'],
            selection_phase=manifest['phase'], validation_score=manifest['validation_score'],
            selection_metric=manifest['metric'], test_sample_acc=final.get('sample_acc'),
            test_window_acc=final.get('window_acc'),
            theory_memory_bytes=theory.get('memory', {}).get('total_bytes'),
            theory_compute_scalars=theory.get('compute', {}).get('total_scalars'),
            theory_access_scalars=theory.get('access', {}).get('total_scalars'),
            theory_time_proxy=theory.get('time_proxy', {}).get('value'), run_id=manifest['run_id']))
    destination = Path(out_dir) / 'final_test_results.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return str(destination)
