def protein(sequence):
    len_seq = len(sequence)
    k_index = sequence.find('K')

    if k_index == -1:
        raise ValueError("Input sequence must contain 'K' (lysine).")

    # 计算31-mer的起始和结束位置(K居中)
    start = k_index - 15
    end = k_index + 16

    # 计算两端需要填充的'X'数量
    left_pad = max(0, -start)  # 左侧不足部分
    right_pad = max(0, end - len_seq)  # 右侧不足部分

    # 调整实际截取范围(不超出原序列边界)
    start = max(0, start)
    end = min(len_seq, end)

    # 获取中心部分并添加填充
    central_part = sequence[start:end]
    padded_sequence = ('X' * left_pad) + central_part + ('X' * right_pad)

    # 确保最终长度为31且K在正中间(索引15)
    if len(padded_sequence) != 31 or padded_sequence[15] != 'K':
        raise ValueError("Sequence processing failed - K not centered correctly.")

    return padded_sequence