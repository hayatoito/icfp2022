API_KEY="XXXXX"

download_problems() {
  cd ./task/problem
  for i in $(seq 1 40); do
    curl -O https://cdn.robovinci.xyz/imageframes/${i}.png
  done
  for i in $(seq 26 40); do
    curl -O https://cdn.robovinci.xyz/imageframes/${i}.initial.json
    curl -O https://cdn.robovinci.xyz/imageframes/${i}.initial.png
  done
}

dump_problem() {
  RUST_LOG=debug cargo run --bin icfp2022 dump-problem ${1:-1}
}

solve() {
  cargo build --release
  time RUST_LOG=info ./target/release/icfp2022 solve ${1:-1}
}

solve_all() {
  cargo build --release
  for i in $(seq 1 35); do
    RUST_LOG=info ./target/release/icfp2022 solve ${i}
  done
}

solve_all_parallel() {
  cargo build --release
  # RUST_LOG=info LANG=C parallel --joblog ./task/log/joblog --results ./task/log/results ./target/release/icfp2022 solve ::: $(seq 1 40)
  # RUST_LOG=info LANG=C parallel --joblog ./task/log/joblog --results ./task/log/results ./target/release/icfp2022 solve ::: $(seq 1 35)
  # RUST_LOG=info LANG=C parallel --joblog ./task/log/joblog --results ./task/log/results ./target/release/icfp2022 solve ::: $(seq 1 6)
}

submit_all() {
  for i in $(seq 1 40); do
    if [[ -f ./task/best/${i}.txt ]]; then
      if [[ -f ./task/submitted/${i}.txt ]] && cmp --silent ./task/best/${i}.txt ./task/submitted/${i}.txt ; then
        echo "Same file exits in ./task/best/${i}.txt. Removing. ./task/best/${i}.txt"
        rm ./task/best/${i}.txt
      else
        curl --header "Authorization: Bearer ${API_KEY}" -F file=@./task/best/${i}.txt https://robovinci.xyz/api/problems/${i}
        mv ./task/best/${i}.txt ./task/submitted/
        sleep 5
      fi
    fi
  done

  fetch_result
}

fetch_result() {
  curl --header "Authorization: Bearer ${API_KEY}" https://robovinci.xyz/api/results/user | tee ./task/result.json | jq .
}

profiling() {
  cargo build --release
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so CPUPROFILE=gperf-cpu.prof ./target/release/icfp2022 solve 7
}

profiling_web() {
  pprof -http=:8080 ./target/release/icfp2022 ./gperf-cpu.prof
}
