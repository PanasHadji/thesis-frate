module "clients" {
  source = "../stef-clients"

  kc_realm_id = var.kc_realm_id
  root_url  = "http://localhost:24006"
  app_client_secret = "dev1234"
}

variable "kc_realm_id" {
  type = string
}
