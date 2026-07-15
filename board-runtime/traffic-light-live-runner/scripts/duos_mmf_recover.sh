#!/bin/sh
set -eu

log() {
  printf '[duos-mmf-recover] %s\n' "$*"
}

show_state() {
  log '===== /proc/cvitek/vpss ====='
  sed -n '1,220p' /proc/cvitek/vpss 2>/dev/null || true
  log '===== /proc/cvitek/sys ====='
  sed -n '1,220p' /proc/cvitek/sys 2>/dev/null || true
}

kill_targets() {
  killall traffic_light_live_runner traffic_light_live_runner_dbg gdb \
    sample_vi_fd sample_vi_od sample_stream_person_capture 2>/dev/null || true
  pkill -f /mnt/data/duos_fd_rtsp_supervisor.sh 2>/dev/null || true
  sleep 1
}

unload_if_present() {
  mod="$1"
  if grep -q "^${mod} " /proc/modules 2>/dev/null; then
    rmmod "$mod" >/dev/null 2>&1 || true
  fi
}

reload_mmf_modules() {
  for mod in \
    cv181x_ive cvi_vc_driver cv181x_jpeg cv181x_vcodec cv181x_tpu \
    cv181x_clock_cooling cv181x_rgn cv181x_mipi_tx cv181x_vo cv181x_dwa \
    cv181x_vpss cv181x_vi snsr_i2c cvi_mipi_rx cv181x_fast_image \
    cv181x_rtos_cmdqu cv181x_base cv181x_sys
  do
    unload_if_present "$mod"
  done

  if [ -f /mnt/system/ko/loadsystemko.sh ]; then
    sh /mnt/system/ko/loadsystemko.sh >/dev/null 2>&1
  else
    log 'loadsystemko.sh not found under /mnt/system/ko'
    return 1
  fi
  sleep 1
}

log 'starting recovery'
show_state
kill_targets
reload_mmf_modules
show_state
log 'recovery complete'
