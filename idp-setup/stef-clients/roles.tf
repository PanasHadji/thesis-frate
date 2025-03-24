locals {
  roles = [
    "SystemAdmin",
  ]
}

resource "keycloak_role" "stef_roles" {
  for_each = toset(local.roles)

  realm_id  = var.kc_realm_id
  client_id = keycloak_openid_client.stef.id

  name = each.key
}
