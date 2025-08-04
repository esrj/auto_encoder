import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

def mapping_height(
    params,
    resolution_x_mm=1.0,
    resolution_y_mm=1.0,
):
    (top_width, bottom_width, height,
     length, tooth_depth, tooth_length, tooth_width,
     groove_length, groove_width, extension_factor) = params

    pitch_x = tooth_length + groove_length
    pitch_y = tooth_width + groove_width

    half_len = length / 2.0
    half_w = bottom_width / 2.0

    # 建立 grid（包含邊界）
    x = np.arange(-half_len, half_len + 1e-8, resolution_x_mm)
    y = np.arange(-half_w, half_w + 1e-8, resolution_y_mm)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # baseline trapezoidal cross-section in Y direction
    Y_abs = np.abs(Y)
    plateau_half = top_width / 2.0
    full_half = half_w

    P = np.zeros_like(Y, dtype=float)
    mask_flat = Y_abs <= plateau_half
    P[mask_flat] = 1.0

    mask_ramp = (Y_abs > plateau_half) & (Y_abs <= full_half)
    if plateau_half < full_half:
        P[mask_ramp] = 1.0 - ((Y_abs[mask_ramp] - plateau_half) / (full_half - plateau_half))

    H = height * P  # baseline height field

    # 只在 top plane (|y| <= plateau_half) 放 tooth intrusion
    top_plane_mask = Y_abs <= plateau_half

    # extension along x（從中心延伸）
    x_ext_half = half_len * extension_factor

    # 對稱 pattern：用 abs
    Ux = np.abs(X)
    Uy = np.abs(Y)

    # x 方向：先 half-groove，再 tooth（shift half groove length）
    half_gx = groove_length / 2.0
    tooth_cycle_pos_x = (Ux - half_gx) % pitch_x  # 這樣第一個 tooth 從 gap/2 開始
    in_tooth_cycle_x = tooth_cycle_pos_x < tooth_length

    # 這個 tooth 的起點（abs 距離）
    tooth_start_x = Ux - tooth_cycle_pos_x
    valid_full_tooth_x = (tooth_start_x + tooth_length) <= x_ext_half  # 整個 tooth 必須在 extension 裡

    # y 方向同理：先 half-groove，再 tooth，且整個 tooth 要在 flat top region 內
    half_gy = groove_width / 2.0
    tooth_cycle_pos_y = (Uy - half_gy) % pitch_y
    in_tooth_cycle_y = tooth_cycle_pos_y < tooth_width
    tooth_start_y = Uy - tooth_cycle_pos_y
    valid_full_tooth_y = (tooth_start_y + tooth_width) <= plateau_half  # 整個 tooth 要在 top flat region

    # 最終 tooth mask（四象限對稱 + full-fit + 只在 top plane）
    tooth_mask = (
        in_tooth_cycle_x & valid_full_tooth_x &
        in_tooth_cycle_y & valid_full_tooth_y &
        top_plane_mask
    )

    H[tooth_mask] -= tooth_depth

    return H  # shape (Nx, Ny)


def padding_height(H):

    H = np.asarray(H)
    if H.ndim != 2:
        raise ValueError(f"padding_height 只接受 2D array，但收到 ndim={H.ndim}")
    target = 400
    h1, h2 = H.shape


    # 如果過大則從中心 crop
    if h1 > target:
        print(" h1 超出 400")
        start_x = (h1 - target) // 2
        H = H[start_x:start_x + target, :]
        h1 = target
    if h2 > target:
        print("h2 超出 400")
        start_y = (h2 - target) // 2
        H = H[:, start_y:start_y + target]
        h2 = target

    # 計算要 pad 的上下左右
    pad_top = (target - h1) // 2
    pad_bottom = target - h1 - pad_top
    pad_left = (target - h2) // 2
    pad_right = target - h2 - pad_left

    H_padded = np.pad(
        H,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    return H_padded


# def plot_height(H, x=None, y=None, filename="img/height_contour_clean.png", levels=50):
#     if x is None:
#         x = np.arange(H.shape[0])
#     if y is None:
#         y = np.arange(H.shape[1])
#     X, Y = np.meshgrid(x, y, indexing="ij")
#
#     fig, ax = plt.subplots()
#     cf = ax.contourf(X, Y, H, levels=levels, cmap="viridis", antialiased=True)
#
#     # 去掉軸的 ticks 和 spines（框線）
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
#     # colorbar 但去掉外框與刻度/label
#     cb = fig.colorbar(cf, shrink=0.6)
#     cb.outline.set_visible(False)  # 去掉 colorbar 外框
#     cb.set_ticks([])
#     cb.set_label("")
#
#     ax.set_aspect("equal", "box")
#     # 讓圖貼滿，不留邊
#     fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0)
#
#     plt.close(fig)


# params = [
#     300.12855863265216,
#     350.47038467955954,
#     69.38178340595373,
#     393.52518303233535,
#     8.938276711111243,
#     19.94502881810993,
#     50.0404329293905,
#     6.28798784423826,
#     30.79488577772473,
#     1.1084625907021999
# ]
#
# H = mapping_height(params, resolution_x_mm=1.0, resolution_y_mm=1.0)
# H = padding_height(H)
# plot_height(H)