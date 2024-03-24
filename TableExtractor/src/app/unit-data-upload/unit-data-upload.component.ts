import { AfterViewInit, Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild } from '@angular/core';
import { FileTransferService } from '../app-services';

@Component({
  selector: 'app-unit-data-upload',
  templateUrl: './unit-data-upload.component.html',
  styleUrls: ['./unit-data-upload.component.css']
})
export class UnitDataUploadComponent implements OnChanges {
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
    this.clearFileInput()
  }

  clearFileInput() {
    if (this.fileInput1 && this.fileInput1.nativeElement) {
      this.fileInput1.nativeElement.value = '';
    }
    if (this.fileInput2 && this.fileInput2.nativeElement) {
      this.fileInput2.nativeElement.value = '';
    }
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
