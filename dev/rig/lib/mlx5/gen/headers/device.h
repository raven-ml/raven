/*
 * Copyright (c) 2013-2015, Mellanox Technologies. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Excerpt of https://raw.githubusercontent.com/torvalds/linux/v6.12/include/linux/mlx5/device.h. */

enum {
	MLX5_BFREGS_PER_UAR		= 4,
	MLX5_MAX_UARS			= 1 << 8,
	MLX5_NON_FP_BFREGS_PER_UAR	= 2,
	MLX5_FP_BFREGS_PER_UAR		= MLX5_BFREGS_PER_UAR -
					  MLX5_NON_FP_BFREGS_PER_UAR,
	MLX5_MAX_BFREGS			= MLX5_MAX_UARS *
					  MLX5_NON_FP_BFREGS_PER_UAR,
	MLX5_UARS_IN_PAGE		= PAGE_SIZE / MLX5_ADAPTER_PAGE_SIZE,
	MLX5_NON_FP_BFREGS_IN_PAGE	= MLX5_NON_FP_BFREGS_PER_UAR * MLX5_UARS_IN_PAGE,
	MLX5_MIN_DYN_BFREGS		= 512,
	MLX5_MAX_DYN_BFREGS		= 1024,
};
