
# def LPV_training_matrix (lpv_mpc_eefig, training_data, threshold, debug=True):

#     for cluster_idx in training_data:

#         matrixes_A = list() # save all values
#         matrixes_B = list() # save all values

#         weights = []

#         # Find All Matrices & Weights
#         for data_set in training_data[cluster_idx]:

#             if data_set.shape[0] >= threshold:

#                 psik = data_set[:-1, :] # only old data
#                 xr = data_set[1:, 0:lpv_mpc_eefig.nx] # xr contains the states x of the buffer (eq. 24)

#                 A, B = lpv_mpc_eefig.create_using_WLS(cluster_idx, xr, psik)
#             else:
#                 continue

#             # Save All Matrixes (WLS)
#             matrixes_A.append(A)
#             matrixes_B.append(B)
#             weights.append(data_set.shape[0])

#         # Security Step
#         if len(matrixes_A) == 0 or len(matrixes_B) == 0:
#             print("WARNING: Training for Granule/Cluster number {} FAILED".format(cluster_idx))
#             continue

#         # Average Matrices
#         weights = np.array(weights)
#         weights = weights / np.sum(weights) # normalize weights

#         A = np.zeros(matrixes_A[0].shape)
#         B = np.zeros(matrixes_B[0].shape)

#         for i in range(len(weights)):
#             A += weights[i] * matrixes_A[i]
#             B += weights[i] * matrixes_B[i]

#         lpv_mpc_eefig.EEFIG[cluster_idx].A = A
#         lpv_mpc_eefig.EEFIG[cluster_idx].B = B


# Create a Granule's A and B (LPV Model) using Window Least Square (WLS)
def create_using_WLS (xr, psik):

    theta = np.zeros([5, 3]) # shell for creating theta in the for loop

    pseudo_inv = np.linalg.pinv(psik)
    for j in range(3):
        theta[:,j] = pseudo_inv @ xr[:, j] # eq. 24 - (2)

    # Update A & B with theta
    A = theta.T[:, :3]
    B = theta.T[:, 3:]

    # Optional Return
    return A, B


def wrong (highest, dict_dif, length):

    for i in range(len(highest)):
        for j in range(i+1, len(highest)):
            if abs(highest[j] - highest[i]) < 100:
                if dict_dif[highest[i]] > dict_dif[highest[j]]:
                    return highest[j]
                else:
                    return highest[i]

            # end
            elif length - highest[j] < 100:
                return highest[j]

            elif length - highest[i] < 100:
                return highest[i]

            # beginning
            elif highest[j] < 100:
                return highest[j]

            elif highest[i] < 100:
                return highest[i]

    return -1


def train2 (lpv_mpc_eefig, data, threshold):

    np.set_printoptions(precision=2, suppress=True)

    init_gran = 12

    matrices_A = list() # save all values
    matrices_B = list() # save all values

    for i in range(0, len(data) - threshold):

        cluster = data[i:i+threshold+1]

        psik = cluster[:-1, :] # only old data
        xr = cluster[1:, 0:3] # xr contains the states x of the buffer (eq. 24)

        A, B = create_using_WLS(xr, psik)

        # Save All Matrixes (WLS)
        matrices_A.append(A)
        matrices_B.append(B)

    dif_A = []
    dif_B = []
    for i in range(len(matrices_A)-1):
        dif_A.append(np.sum(matrices_A[i+1]-matrices_A[i]))
        dif_B.append(np.sum(matrices_B[i+1]-matrices_B[i]))

    dif_A = np.array(dif_A)
    dif_B = np.array(dif_B)

    dif = abs(dif_A) + abs(dif_B)






    # derivar 
    # dif = abs(np.gradient(dif))

    dict_dif = {}
    for i in range(dif.shape[0]):
        dict_dif[i] = dif[i]

    validated = False
    while not validated:

        highest = nlargest(init_gran - 1, dict_dif, key = dict_dif.get)
        idx = wrong(highest, dict_dif, dif.shape[0])

        if idx < 0:
            validated = True
        else:
            del(dict_dif[idx])

    plt.plot(dif)
    for val in highest:
        plt.axvline(x=val, color='red', linestyle='--')
    plt.show()




    # Normalize data
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)

    data_normalized = data - min_
    data_normalized /= (max_ - min_)



    cmapf = get_cmap(init_gran + 1) # color

    idx = 0
    cmap = []
    for i in range(data.shape[0]):
        if i in highest:
            idx += 1
        cmap.append(cmapf(idx))
    
    fig = plt.figure(figsize=plt.figaspect(0.3333333333333))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(data_normalized[:, 0], data_normalized[:, 1], data_normalized[:, 2], color= cmap, s=20, alpha=0.5)
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(data_normalized[:, 1], data_normalized[:, 2], data_normalized[:, 3], color= cmap, s=20, alpha=0.5)
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(data_normalized[:, 2], data_normalized[:, 3], data_normalized[:, 4], color= cmap, s=20, alpha=0.5)
    
    plt.title("3D Representation of Best Cluster Selected")
    plt.show()













    highest.sort()

    mat_A = []
    mat_B = []
    cov   = []
    m     = []
    w     = []
    idxs  = []

    start   = 0
    end     = highest[0]

    length = end - start
    idxs.append([(start, end)])

    w.append(length)

    A = sum(matrices_A[start:end]) / length
    B = sum(matrices_B[start:end]) / length

    mat_A.append(A)
    mat_B.append(B)

    cov_ = np.cov(data[start:end, :], rowvar=False)
    cov.append(cov_)
    m_ = np.mean(data[start:end, :], axis = 0)
    m.append(m_)

    for i in range(1, len(highest)):
        start   = highest[i - 1]
        end     = highest[i]
        length = end - start

        w.append(length)
        idxs.append([(start, end)])

        A = sum(matrices_A[start:end]) / length
        B = sum(matrices_B[start:end]) / length

        mat_A.append(A)
        mat_B.append(B)

        cov_ = np.cov(data[start:end, :], rowvar=False)
        cov.append(cov_)
        m_ = np.mean(data[start:end, :], axis = 0)
        m.append(m_)

    start   = highest[-1]
    end     = dif.shape[0]

    length = end - start
    idxs.append([(start, end)])

    w.append(length)

    A = sum(matrices_A[start:end]) / length
    B = sum(matrices_B[start:end]) / length

    mat_A.append(A)
    mat_B.append(B)

    cov_ = np.cov(data[start:end, :], rowvar=False)
    cov.append(cov_)
    m_ = np.mean(data[start:end, :], axis = 0)
    m.append(m_)

    m_array = np.array(m)

    print(cov)
    print(m_array)



    fig = plt.figure(figsize=plt.figaspect(0.333333333))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(m_array[:, 0], m_array[:, 1], m_array[:, 2], s=20, alpha=0.5)
    
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.scatter(m_array[:, 1], m_array[:, 2], m_array[:, 3], s=20, alpha=0.5)
    
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.scatter(m_array[:, 2], m_array[:, 3], m_array[:, 4], s=20, alpha=0.5)

    plt.show()
    

    dif_matA    = np.zeros([init_gran,init_gran])
    dif_matB    = np.zeros([init_gran,init_gran])
    dif_m       = np.zeros([init_gran,init_gran])

    for i in range(len(mat_A)):
        for j in range(len(mat_A)):
            dA = np.sum(abs(mat_A[i] - mat_A[j]))
            dif_matA[i,j] = dA

            dB = np.sum(abs(mat_B[i] - mat_B[j]))
            dif_matB[i,j] = dB

            dm = np.sum(abs(m[i] - m[j]))
            dif_m[i,j] = dm

    print()
    print(dif_matA)
    print()
    print(dif_matB)
    print()
    print(dif_m)
    print()

    dif_t = dif_m + dif_matA + dif_matB
    print(dif_t)
    print()

    # Join similar granules
    np.fill_diagonal(dif_t, 1e32)

    x, y = np.where(dif_t == np.min(dif_t)) # simetrical matrix
    x = x[0]
    y = y[0]

    while dif_t[x, y] < 4:

        # add granule x & y
        mat_A[x]    = (mat_A[x]*w[x] + mat_A[y]*w[y]) / (w[x] + w[y])
        mat_B[x]    = (mat_B[x]*w[x] + mat_B[y]*w[y]) / (w[x] + w[y])
        w[x]        = w[x] + w[y]
        m[x]        = (m[x]*w[x] + m[y]*w[y]) / (w[x] + w[y])
        cov[x]      = (cov[x]*w[x] + cov[y]*w[y]) / (w[x] + w[y])
        idxs[x]     = idxs[x] + idxs[y]

        mat_A.pop(y)
        mat_B.pop(y)
        w.pop(y)
        m.pop(y)
        cov.pop(y)
        idxs.pop(y)

        dif_matA    = np.zeros([len(mat_A),len(mat_A)])
        dif_matB    = np.zeros([len(mat_A),len(mat_A)])
        dif_m       = np.zeros([len(mat_A),len(mat_A)])

        for i in range(len(mat_A)):
            for j in range(len(mat_A)):
                dA = np.sum(abs(mat_A[i] - mat_A[j]))
                dif_matA[i,j] = dA

                dB = np.sum(abs(mat_B[i] - mat_B[j]))
                dif_matB[i,j] = dB

                dm = np.sum(abs(m[i] - m[j]))
                dif_m[i,j] = dm

        dif_t = dif_m + dif_matA + dif_matB


        np.fill_diagonal(dif_t, 1e16)
        print(np.min(dif_t))

        x, y = np.where(dif_t == np.min(dif_t)) # simetrical matrix
        x = x[0]
        y = y[0]


        print(dif_t)

    m_array = np.array(m)


    fig = plt.figure(figsize=plt.figaspect(0.333333333))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(m_array[:, 0], m_array[:, 1], m_array[:, 2], s=20, alpha=0.5)
    
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.scatter(m_array[:, 1], m_array[:, 2], m_array[:, 3], s=20, alpha=0.5)
    
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.scatter(m_array[:, 2], m_array[:, 3], m_array[:, 4], s=20, alpha=0.5)

    plt.show()



    datasets = []

    granules = []
    # Generate Granule With Cluster Points
    for i in range(len(mat_A)):
        
        dataset = data[idxs[i][0][0]:idxs[i][0][1]]
        for j in range(1, len(idxs[i])):
            print(idxs[i])
            dataset = np.vstack([dataset, data[idxs[i][j][0]:idxs[i][j][1]]])

        datasets.append(dataset)

        granule = Granule(lpv_mpc_eefig.p, dataset)

        granule.A = mat_A[i]
        granule.B = mat_B[i]

        # Save Granule
        granules.append(granule)

    # Change EEFIG List For New List
    lpv_mpc_eefig.set_granules(granules)

    cmapf = get_cmap(len(mat_A) + 1) # color

    cmap = [0] * (len(data) - threshold -1)
    for i in range(len(idxs)):
        for start, end in idxs[i]:
            cmap[start:end] = [cmapf(i)] * (end-start)
    
    datasets_ = datasets[0]
    for i in range(1, len(datasets)):
        datasets_ = np.vstack([datasets_, datasets[i]])

    # Normalize data
    min_ = np.min(datasets_, axis=0)
    max_ = np.max(datasets_, axis=0)

    datasets_normalized = datasets_ - min_
    datasets_normalized /= (max_ - min_)


    fig = plt.figure(figsize=plt.figaspect(0.3333333333333))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(datasets_normalized[:, 0], datasets_normalized[:, 1], datasets_normalized[:, 2], color= cmap, s=20, alpha=0.5)
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(datasets_normalized[:, 1], datasets_normalized[:, 2], datasets_normalized[:, 3], color= cmap, s=20, alpha=0.5)
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(datasets_normalized[:, 2], datasets_normalized[:, 3], datasets_normalized[:, 4], color= cmap, s=20, alpha=0.5)
    
    plt.title("3D Representation of Best Cluster Selected")
    plt.show()





    return lpv_mpc_eefig