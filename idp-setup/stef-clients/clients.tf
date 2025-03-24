resource "keycloak_openid_client" "frate" {
  realm_id  = var.kc_realm_id
  client_id = "frate"

  name        = "FRATE"
  access_type = "CONFIDENTIAL"

  standard_flow_enabled = true

  client_secret = var.app_client_secret

  root_url                        = var.root_url
  valid_redirect_uris             = ["http://localhost:24006/*"]
  valid_post_logout_redirect_uris = ["http://localhost:24006/*"]
  web_origins = ["+"]
  implicit_flow_enabled = true

  frontchannel_logout_enabled = true
}

resource "keycloak_openid_user_client_role_protocol_mapper" "frate_roles" {
  realm_id  = var.kc_realm_id
  client_id = keycloak_openid_client.frate.id
  name      = "frate-roles"

  claim_name                  = "resource_access.$${client_id}.roles"
  client_id_for_role_mappings = keycloak_openid_client.frate.client_id
  multivalued                 = true

  add_to_access_token = true
  add_to_id_token     = true
  add_to_userinfo     = false
}


resource "keycloak_openid_client_default_scopes" "frate_scopes_default" {
  realm_id  = var.kc_realm_id
  client_id = keycloak_openid_client.frate.id

  default_scopes = [
    "profile",
    "email",
  ]
}

resource "keycloak_openid_client_optional_scopes" "frate_scopes_optional" {
  realm_id  = var.kc_realm_id
  client_id = keycloak_openid_client.frate.id

  optional_scopes = []
}
