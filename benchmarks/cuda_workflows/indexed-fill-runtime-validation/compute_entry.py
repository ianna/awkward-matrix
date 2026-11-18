def awkward_IndexedArray_fill(toindex, toindexoffset, fromindex, length, base):
    from awkward._connect.cuda._indexed_fill import fill

    fill(toindex, toindexoffset, fromindex, length, base)
