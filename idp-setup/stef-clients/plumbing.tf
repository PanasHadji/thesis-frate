terraform {
  required_providers {
    keycloak = {
      source  = "mrparkers/keycloak"
      version = "~> 4.3"
    }
  }
}

variable "kc_realm_id" {
  type = string
}

variable "root_url" {
  type = string
}

variable "app_client_secret" {
  type = string

  nullable = true
  default = null

  sensitive = true
}
