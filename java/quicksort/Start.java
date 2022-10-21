package quicksort;

import mpi.Cartcomm;
import mpi.MPI;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Start {
    public static void main(String[] args) {
        MPI.Init(args);
        long time = 0;
        time -= System.currentTimeMillis();
        final int numberOfElements = (int) (Math.pow((14 - 5), 2) * 1000);
        int[] chunk;
        int numberOfProcess = MPI.COMM_WORLD.Size();
        int rankOfProcess = MPI.COMM_WORLD.Rank();
        int[] dims = new int[]{numberOfProcess};
        boolean[] periods = new boolean[]{true};
        Cartcomm circle = MPI.COMM_WORLD.Create_cart(dims, periods, true);
        System.out.println("Process with rank: " + rankOfProcess + " starting his job");
        int chunkSize = (numberOfElements % numberOfProcess == 0) ? (numberOfElements / numberOfProcess) : (numberOfElements / (numberOfProcess - 1));
        int[] data = new int[numberOfProcess * chunkSize];

        circle.Barrier();

        if (rankOfProcess == 0) {
            System.out.println("Work began...");
            System.out.println("Elements required - " + numberOfElements);
            System.out.println("Every process will sort an equal part, which contain - " + chunkSize + " elements");
            try (FileWriter fw = new FileWriter("MyArrays/unsorted.txt")) {
                for (int i = 0; i < numberOfElements; i++) {
                    data[i] = new Random().nextInt(0, 1001);
                    if (i % 30 == 0)
                        fw.write(data[i] + ",\n");
                    fw.write(data[i] + ", ");
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            for (int i = numberOfElements; i < numberOfProcess * chunkSize; i++) {
                data[i] = 0;
            }
        }
        circle.Barrier();

        System.out.println("Working...");
        int[] buffer = {numberOfElements};
        circle.Bcast(buffer, 0, 1, MPI.INT, 0);
        chunk = new int[chunkSize];

        circle.Barrier();

        circle.Scatter(data, 0, chunkSize, MPI.INT, chunk, 0, chunkSize, MPI.INT, 0);

        circle.Barrier();

        int ownChunkSize = (numberOfElements >= chunkSize * (rankOfProcess + 1)) ? chunkSize : (numberOfElements - chunkSize * rankOfProcess);
        quicksort(chunk, 0, ownChunkSize);

        circle.Barrier();

        for (int step = 1; step < numberOfProcess; step = 2 * step) {
            if (rankOfProcess % (2 * step) != 0) {
                circle.Send(chunk, 0, ownChunkSize, MPI.INT, rankOfProcess - step, 0);
                break;
            }
            if (rankOfProcess + step < numberOfProcess) {
                int receivedChunkSize = (numberOfElements >= chunkSize * (rankOfProcess + 2 * step)) ? (chunkSize * step) : (numberOfElements - chunkSize * (rankOfProcess + step));
                int[] chunkReceived = new int[receivedChunkSize];
                circle.Recv(chunkReceived, 0, receivedChunkSize, MPI.INT, rankOfProcess + step, 0);
                data = merge(chunk, ownChunkSize, chunkReceived, receivedChunkSize);
                chunk = data;
                ownChunkSize = ownChunkSize + receivedChunkSize;
            }
        }

        circle.Barrier();

        time += System.currentTimeMillis();
        if (rankOfProcess == 0) {
            System.out.println("Sorting process ended correctly, you can see the results in generated .txt files");
            try (FileWriter fw = new FileWriter("MyArrays/sorted.txt")) {
                for (int i = 0; i < numberOfElements; i++) {
                    if (i % 30 == 0)
                        fw.write(data[i] + ",\n");
                    fw.write(data[i] + ", ");
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("Total number of elements given as input : " + data.length);
            System.out.println("The number of processors, which were used during sorting: " + numberOfProcess);
            System.out.println("Time spent: " + time + " milliseconds");
        }
        MPI.Finalize();
    }

    static void swap(int[] arr, int i, int j) {
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }

    static void quicksort(int[] arr, int start, int end) {
        int pivot;
        int index;
        if (end <= 1)
            return;
        pivot = arr[start + end / 2];
        swap(arr, start, start + end / 2);
        index = start;
        for (int i = start + 1; i < start + end; i++) {
            if (arr[i] < pivot) {
                index++;
                swap(arr, i, index);
            }
        }
        swap(arr, start, index);
        quicksort(arr, start, index - start);
        quicksort(arr, index + 1, start + end - index - 1);
    }

    static int[] merge(int[] arr1, int n1, int[] arr2, int n2) {
        int[] result = new int[n1 + n2];
        int i = 0;
        int j = 0;
        int k;
        for (k = 0; k < n1 + n2; k++) {
            if (i >= n1) {
                result[k] = arr2[j];
                j++;
            } else if (j >= n2 || arr1[i] < arr2[j]) {
                result[k] = arr1[i];
                i++;
            } else {
                result[k] = arr2[j];
                j++;
            }
        }
        return result;
    }
}
