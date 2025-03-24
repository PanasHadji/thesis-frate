locals {
  roles = [
    "SystemAdmin",
  ]
}

resource "keycloak_role" "frate_roles" {
  for_each = toset(local.roles)

  realm_id  = var.kc_realm_id
  client_id = keycloak_openid_client.frate.id

  name = each.key
}
