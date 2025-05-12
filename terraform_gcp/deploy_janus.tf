provider "google" {
  project = "stocks-451219"
  region  = "us-central1"
  zone    = "us-central1-a"
}

# Create a custom VPC network
resource "google_compute_network" "vpc_network" {
  name                    = "trading-app-network"
  auto_create_subnetworks = "true"
}

# Firewall rule to allow traffic to your container
resource "google_compute_firewall" "allow_trading_app" {
  name    = "allow-trading-app"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8080", "4001", "4003"]  # Updated to use port 4001 instead of 4004
  }

  source_ranges = ["0.0.0.0/0"]  # Consider restricting this for production

  # Add explicit dependency
  depends_on = [google_compute_network.vpc_network]
}

# Create a static IP for the VM to ensure the scheduler can always reach it
resource "google_compute_address" "static_ip" {
  name = "trading-app-ip"
}

# Create the VM instance with properly configured container
resource "google_compute_instance" "trading_app_vm" {
  name         = "trading-app-vm"
  machine_type = "e2-medium"
  
  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 20
    }
  }

  network_interface {
    network = google_compute_network.vpc_network.name
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  # Service account for pulling from Artifact Registry
  service_account {
    email  = "stocks@stocks-451219.iam.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }

  # Updated metadata with properly configured container declaration
  # IB_PORT is consumed by the web server
  metadata = {
    gce-container-declaration = <<EOF
spec:
  containers:
    - name: trading-app
      image: us-central1-docker.pkg.dev/stocks-451219/janus-repo/janus:future
      env:
        - name: TWS_USERID
          value: "garciaj42"
        - name: TRADING_MODE
          value: "paper"
        - name: READ_ONLY_API
          value: "no"
        - name: TWOFA_TIMEOUT_ACTION
          value: "restart"
        - name: AUTO_RESTART_TIME
          value: "11:59 PM"
        - name: RELOGIN_AFTER_TWOFA_TIMEOUT
          value: "yes"
        - name: TIME_ZONE
          value: "America/Los_Angeles"
        - name: ACCEPT_INCOMING_CONNECTION
          value: "yes"
        - name: TRUSTED_IPS
          value: "0.0.0.0/0"
        - name: IB_PORT
          value: "4004"
        - name: IB_CLIENT_ID
          value: "7"
        - name: TWS_PASSWORD
          value: "${data.google_secret_manager_secret_version.tws_password.secret_data}"
      ports:
        - containerPort: 8080
          hostPort: 8080
      stdin: false
      tty: false
      restartPolicy: Always
  restartPolicy: Always
EOF

    user-data = <<EOF
#cloud-config
runcmd:
  - sudo chmod 666 /var/run/docker.sock
  - gcloud auth configure-docker us-central1-docker.pkg.dev
EOF
  }

  # Allow the instance to connect to GCP services
  tags = ["trading-app"]
}

# Reference existing secret instead of creating a new one
data "google_secret_manager_secret" "tws_password" {
  secret_id = "tws_password"
  project   = "stocks-451219"
}

data "google_secret_manager_secret_version" "tws_password" {
  secret = data.google_secret_manager_secret.tws_password.id
  version = "latest"
}

# Cloud Scheduler job to call 8080/daily_futures at 7am daily
resource "google_cloud_scheduler_job" "daily_futures_job" {
  name        = "daily-futures-job"
  description = "Calls the trading app's daily_futures endpoint at 7am daily"
  schedule    = "0 7 * * *"
  time_zone   = "America/Los_Angeles"
  
  http_target {
    uri         = "http://${google_compute_address.static_ip.address}:8080/daily_futures"
    http_method = "GET"
  }
}

# Updated health check to check the IB Gateway
resource "google_compute_health_check" "trading_app_health_check" {
  name                = "trading-app-health-check"
  check_interval_sec  = 60
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port = 4001  # Use the socat port
    request_path = "/status"
  }
}

# Create an uptime check to monitor your endpoint
resource "google_monitoring_uptime_check_config" "trading_app_uptime" {
  display_name = "Trading App Uptime Check"
  timeout      = "10s"
  period       = "300s"
  
  http_check {
    path           = "/status"
    port           = "8080"
    use_ssl        = false
    validate_ssl   = false
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      host = google_compute_address.static_ip.address
    }
  }
}

# Create a notification channel for alerts
resource "google_monitoring_notification_channel" "email" {
  display_name = "Trading App Alerts"
  type         = "email"
  labels = {
    email_address = "your-email@example.com"  # Update with your email
  }
}

# Create an alerting policy for the uptime check
resource "google_monitoring_alert_policy" "uptime_alert" {
  display_name = "Trading App Down Alert"
  combiner     = "OR"
  
  conditions {
    display_name = "Uptime Check Failed"
    condition_threshold {
      filter          = "resource.type = \"uptime_url\" AND metric.type = \"monitoring.googleapis.com/uptime_check/check_passed\""
      duration        = "300s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1
      
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
}

# Allow SSH via IAP tunnel
resource "google_compute_firewall" "allow_ssh_oslogin_from_iap" {
  name        = "allow-ssh-oslogin-from-iap"
  project     = "stocks-451219"
  network     = "trading-app-network"
  description = "Allow SSH via IAP tunnel for OS Login"
  
  direction   = "INGRESS"
  priority    = 1000

  allow {
    protocol  = "tcp"
    ports     = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
}

# Output the external IP for easy access
output "external_ip" {
  value = google_compute_address.static_ip.address
}

# Output the Cloud Scheduler job details
output "scheduler_job" {
  value = google_cloud_scheduler_job.daily_futures_job.name
}