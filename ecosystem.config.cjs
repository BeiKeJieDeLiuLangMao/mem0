module.exports = {
  apps: [
    {
      name: 'mem0',
      cwd: '/Users/yishu.cy/IdeaProjects/openclaw-team-workspace/mem0',
      script: 'python3',
      args: 'server.py',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1'
      },
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      exp_backoff_restart_delay: 100,
      // macOS specific
      out_file: '/tmp/mem0_stdout.log',
      error_file: '/tmp/mem0_stderr.log',
      log_file: '/tmp/mem0_combined.log',
      time: true
    }
  ]
};
