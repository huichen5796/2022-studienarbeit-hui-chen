import { AfterViewInit, Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild } from '@angular/core';
import { FileTransferService } from '../app-services';

@Component({
  selector: 'app-unit-data-upload',
  templateUrl: './unit-data-upload.component.html',
  styleUrls: ['./unit-data-upload.component.css']
})
export class UnitDataUploadComponent implements AfterViewInit, OnChanges {
  @ViewChild('fileInput1') fileInput1!: ElementRef;
  @ViewChild('fileInput2') fileInput2!: ElementRef;

  @Input() onlyDir!: boolean;
  @Input() allowMultiple!: boolean;
  @Input() allowFolder!: boolean;

  dataStoreStatus: 'error' | 'success' | 'pending' = 'pending'

  selectedFiles: { [key: string]: string } = {}

  constructor(
    private fileTransferService: FileTransferService,
  ) {

  }
  ngOnChanges(changes: SimpleChanges): void {
    if (!changes['data_type'].isFirstChange()) {
      this.ngAfterViewInit()
    }
  }

  clearFileInput() {
    if (this.fileInput1 && this.fileInput1.nativeElement) {
      this.fileInput1.nativeElement.value = '';
    }
    if (this.fileInput2 && this.fileInput2.nativeElement) {
      this.fileInput2.nativeElement.value = '';
    }
  }

  ngAfterViewInit() {
    // this.connectTestService.cleanDataStoreFile(this.data_type).then((rep) => {
    //   this.dataStoreStatus = rep
    //   this.uploadedFiles = []
    //   this.selectedFiles = {};
    //   this.clearFileInput()
    // })
    this.clearFileInput()
  }

  onFileSelected(event: any) {
    const files: FileList = event.target.files;
    if (files.length > 0) {
      this.fileTransferService.fileUpload(files).then((res) => {
        console.log(res)

      })
    }
  }
}
