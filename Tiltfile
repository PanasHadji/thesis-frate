# Enforce a minimum Tilt version
# https://docs.tilt.dev/api.html#api.version_settings
version_settings(constraint='>=0.22.1')

load('ext://pack', 'pack')

def kn_func_node(name):
  func_dir = 'node-funcs/' + name

  pack(
    'ghcr.io/inspire-research-center/stef-' + name,
    # https://github.com/knative/func/blob/527ef6cce609123717adf080e4c7a9cc09fcbe38/pkg/builders/buildpacks/builder.go#L30
    builder='ghcr.io/knative/builder-jammy-base:0.4.283',
    path=func_dir,
    deps=[func_dir],
    ignore=[
      '.func',
      'func.yaml',
    ],
    live_update = [
      fall_back_on([
        func_dir + '/.funcignore',
        func_dir + '/requirements.txt',
      ]),

      # Sync local files into the container.
      sync(func_dir, '/workspace'),

      # Restart the process to pick up the changed files.
      restart_container(),
    ],
  )

# Value of `name` must be
# * Name of the folder where the function resides
# * Name of the service in docker-compose.yml
# * Name of the container image (after the `ghcr.io/inspire-research-center/stef-` prefix)
# * DNS-compliant (e.g. no underscores, latin characters only, etc.)
# If one of these is not feasible, talk to Artem about setting up a more flexible configuration

kn_func_node('impute-missing-values')
kn_func_node('stats')
kn_func_node('export-csv')
kn_func_node('read-csv')
kn_func_node('decision-tree')
kn_func_node('filter-df')
kn_func_node('drop-columns')
kn_func_node('sort-columns')
kn_func_node('calculate-trust-stats')
kn_func_node('manage-outliers')
kn_func_node('clustering-v2')
kn_func_node('decision-tree-basic')
kn_func_node('correlation-analysis')
kn_func_node('normalization')
kn_func_node('import-text-data')
kn_func_node('import-online-data')
kn_func_node('export-to-file')
kn_func_node('append-tables')
kn_func_node('join-tables')
kn_func_node('nearest-neighbors')
kn_func_node('random-forest')
kn_func_node('linear-svm')
kn_func_node('rbf-svm')
kn_func_node('neural-network')
kn_func_node('ada-boost')
kn_func_node('naive-bayes')
kn_func_node('xg-boost')

docker_compose('docker-compose.yml')
