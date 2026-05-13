#!/bin/bash

while true; do
    # 执行 rsync 命令
    rsync -avzP -e "ssh -F ./ssh_config -p 20022 -c aes128-ctr" swin@cdn.tpami.com:/home/swin/zyli/BackdoorAttack/WeatherApproximationGAN/shapley-adv/main /data/zyli/BackdoorAttack/shapley_advGAN/

    # 检查 rsync 的返回值
    if [ $? -eq 0 ]; then
        echo "Rsync completed successfully!"
        break
    else
        echo "Rsync failed!Retrying..."
    fi
done

exit 0
