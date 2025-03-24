terraform {
  required_providers {
    keycloak = {
      source  = "mrparkers/keycloak"
      version = "~> 4.3"
    }
  }
}

provider "keycloak" {
  url = var.kc_url

  client_id     = var.kc_client_id
  client_secret = var.kc_client_secret

  username = var.kc_username
  password = var.kc_password
}

variable "kc_url" {
  type     = string
  nullable = false
}

variable "kc_client_id" {
  type     = string
  nullable = false
}

variable "kc_client_secret" {
  type    = string
  default = ""
}

variable "kc_username" {
  type    = string
  default = ""
}

variable "kc_password" {
  type    = string
  default = ""
}
